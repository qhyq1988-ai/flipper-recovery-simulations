"""Fit and validate the Flipper models before testing online communication."""
from __future__ import annotations

from dataclasses import asdict, replace
from itertools import combinations, product
from statistics import mean, median
import numpy as np

from identification import fit, load_logs, error_measures
from platform_model import DT, CHANNELS, WARM_TICKS, Scenario, physical_trace, reference, on_board_offsets
from study import write_csv, write_json, G1_SEED, static_run

FORECAST_DELAYS = (0, 25, 40, 50, 55, 65)
PROFILES = ("nominal", "loaded", "asymmetric")


def identification_scene(seed, profile, i):
    r = np.random.default_rng(seed ^ 7277)
    phase = int(r.integers(0, 360))
    sg = 1 if i % 2 == 0 else -1
    return Scenario(f"id_{seed}", "identification", seed, phase, (
        (85, sg*int(r.integers(5, 11))),
        (165, -sg*int(r.integers(5, 11))),
        (250, sg*int(r.integers(3, 9))),
        (375, -sg*int(r.integers(4, 10))),
        (520, 0),
    ), horizon=650, profile=profile)


def save_log(path, s, rr):
    u = rr["plant"]["pwm_angle"]
    q = rr["plant"]["angle"]
    count = rr["plant"]["pwm_tick"]
    r = np.random.default_rng(s.seed ^ 1500766)
    measured = np.round((q + r.normal(0., .065, q.shape)) / .10) * .10
    held = np.r_[np.zeros(WARM_TICKS, dtype=int), rr["held"]]
    rows = []
    for k in range(len(measured)):
        row = {"t_s": k*DT, "task_t_s": (k-WARM_TICKS)*DT, "applied_offset_deg": int(held[k])}
        row.update({f"pwm{j}": float(u[k,j]) for j in range(CHANNELS)})
        row.update({f"q{j}": float(measured[k,j]) for j in range(CHANNELS)})
        row.update({f"pwm_count{j}": int(count[k,j]) for j in range(CHANNELS)})
        rows.append(row)
    write_csv(path, rows)


def inspect_same_final_input():
    s = Scenario("history_check", "diagnosis", G1_SEED+99, 115, ((135, -8), (270, 9)),
                 horizon=450, profile="asymmetric")
    at = 270
    scan = []
    for duration, prev, pulse in product((5,10,15,20), (-10,-5,0,5,10), (-10,-5,0,5,10)):
        plan = {135: -8, 205: prev, at-duration: pulse, at:9}
        held = on_board_offsets(s.horizon, plan)
        world = physical_trace(s, held)
        idx = at + WARM_TICKS
        q, v, gear = [world[k][idx] for k in ("angle","velocity","internal_gear")]
        tail = world["angle"][idx:idx+75]
        scan.append((q, v, gear, tail, (duration, prev, pulse)))
    admissible = []
    for a,b in combinations(scan,2):
        dq, dv, di = [float(np.max(np.abs(a[i]-b[i]))) for i in range(3)]
        if dq <= .70 and dv <= 6.0 and a[-1]!=b[-1]:
            delta = float(np.max(np.abs(a[3][10:] - b[3][10:])))
            admissible.append((delta, dq, dv, di, a[-1], b[-1]))
    selected = max(admissible, default=None)
    return {
        "search_profiles": len(scan),
        "pairs_angle_max_0p7_and_speed_max_6": len(admissible),
        "comparison_start_ms_after_application": 100,
        "report": ("Largest remaining difference once angle and speed are approximately matched; "
                   "residual angle and speed differences remain."),
        "best": dict(zip(("future_angle_separation_deg", "angle_match_deg", "speed_match_deg_s",
                          "latent_backlash_state_difference_deg", "history_a", "history_b"),selected)) if selected else None,
    }


def run_identification(out):
    logs_dir = out / "identification_logs"
    logs_dir.mkdir(exist_ok=True)
    rules = {
        "gate": "Reference course: recovery MAE below one 50-ms opportunity and gain of at least 10 ms "
                "against stateless model without increasing false feasible predictions, on nominal and "
                "on at least one held-out plant profile.",
        "holdout_submission_lags_ms": [t * 10 for t in FORECAST_DELAYS],
        "validation_profiles": PROFILES,
        "model_training_loss": "free rollout RMSE after first second, on nominal identification logs only",
    }
    write_json(out / "g1_protocol.json", rules)
    train = []
    for i in range(9):
        s = identification_scene(G1_SEED+i,"nominal",i)
        r = static_run(s,current=True)
        p = logs_dir / f"fit_{s.seed}.csv"
        save_log(p,s,r)
        train.append(p)
    fit_data = load_logs(train)
    models = {}
    for name, uses_memory in (("stateless",False), ("memory",True)):
        fitted, rmse = fit(fit_data,uses_memory)
        models[name] = fitted, rmse

    all_cases = []
    forecast_traces = []
    for group, profile in enumerate(PROFILES):
        start = G1_SEED + 10 + (0 if group == 0 else 10 + (group-1)*6)
        for i in range(6):
            seed = start + i
            s = identification_scene(seed,profile,i)
            r = static_run(s,current=True)
            save_log(logs_dir / f"holdout_{profile}_{seed}.csv",s,r)
            # All alternatives share the same known prefix and next requested command.
            s = replace(s, events=s.events[:3], horizon=475)
            begin, value = s.events[-1]
            for delay in FORECAST_DELAYS:
                ts = tuple(t for t,_ in s.events[:-1]) + (begin+delay,)
                trial = static_run(s,ts)
                u = trial["plant"]["pwm_angle"]
                truth = trial["plant"]["angle"][WARM_TICKS:]
                desired = reference(s)
                e, t_real, peak_true, actual = error_measures(truth, desired, s.horizon+1, begin)
                for name, (model, _) in models.items():
                    hats = model.run(u)[0][WARM_TICKS:]
                    eh, t_pred, peak_pred, predicts = error_measures(hats, desired, s.horizon+1, begin)
                    penalty = s.horizon-begin+1
                    terr = abs((penalty if t_real is None else t_real) - (penalty if t_pred is None else t_pred))
                    all_cases.append({
                        "profile": profile, "seed": seed, "model": name, "lag_ms": delay*10,
                        "actual_recovery_s": t_real*DT if t_real is not None else None,
                        "predicted_recovery_s": t_pred*DT if t_pred is not None else None,
                        "unresolved_actual": int(t_real is None), "unresolved_predicted": int(t_pred is None),
                        "recovery_error_s": terr*DT, "peak_actual_deg": peak_true, "peak_predicted_deg": peak_pred,
                        "actual_pass": actual, "predicted_pass": predicts,
                        "false_feasible": int(bool(not actual and predicts)),
                        "trajectory_rmse_deg": float(np.sqrt(np.mean((hats[begin:]-truth[begin:])**2))),
                    })
                    if delay==55 and i==0:
                        for k in range(begin-20,s.horizon+1):
                            forecast_traces.append({"profile":profile,"model":name,"t_relative_s":(k-begin)*DT,
                                                    "true_peak_deg":float(e[k]),"pred_peak_deg":float(eh[k])})

    tally = {}
    for profile in PROFILES:
        tally[profile] = {}
        for name in models:
            a = [x for x in all_cases if x["profile"]==profile and x["model"]==name]
            tally[profile][name] = {
                "trials": len(a),
                "rmse_deg": mean(x["trajectory_rmse_deg"] for x in a),
                "mean_recovery_error_s": mean(x["recovery_error_s"] for x in a),
                "median_recovery_error_s": median(x["recovery_error_s"] for x in a),
                "false_feasible": sum(x["false_feasible"] for x in a),
                "actual_failures": sum(1-x["actual_pass"] for x in a),
                "never_recovered": sum(x["unresolved_actual"] for x in a),
            }
    profiles_met = []
    for p,a in tally.items():
        if a["memory"]["mean_recovery_error_s"] < .05 and \
           a["stateless"]["mean_recovery_error_s"] - a["memory"]["mean_recovery_error_s"] >= .010 and \
           a["memory"]["false_feasible"] <= a["stateless"]["false_feasible"]:
            profiles_met.append(p)
    passed = "nominal" in profiles_met and len(profiles_met) > 1
    report = {
        "gate": passed, "fit": {n: {"parameters":asdict(m), "training_rmse_deg":r} for n,(m,r) in models.items()},
        "validation": tally, "profiles_met": profiles_met, "history_pair": inspect_same_final_input(),
    }
    write_csv(out/"g1_validation.csv",all_cases)
    write_csv(out/"g1_examples.csv",forecast_traces)
    write_json(out/"g1_report.json",report)
    return report, {n:m for n,(m,_) in models.items()}

