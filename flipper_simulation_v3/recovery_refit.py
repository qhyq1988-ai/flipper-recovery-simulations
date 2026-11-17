"""One equal-opportunity refit targeting recovery, with untouched confirmation seeds."""
from __future__ import annotations
from dataclasses import asdict, replace
from itertools import product
from statistics import mean
import numpy as np

from identification import Predictor, error_measures
from identification_study import identification_scene
from platform_model import DT, WARM_TICKS, RECOVERY_TICKS, ERROR_BAND, reference
from study import static_run, write_json, write_csv

CALIBRATION_SEEDS = tuple(range(36701, 36707))
CONFIRM_SEEDS = {
    "nominal": tuple(range(36721, 36727)),
    "loaded": tuple(range(36741, 36747)),
    "asymmetric": tuple(range(36761, 36767)),
}
LAGS = (0, 30, 45, 55, 60, 65)


def cases(profile, seeds):
    out = []
    for i, seed in enumerate(seeds):
        full = identification_scene(seed, profile, i)
        sc = replace(full, events=full.events[:3], horizon=475)
        begin = sc.events[-1][0]
        end = sc.horizon + 1
        ref = reference(sc)
        for lag in LAGS:
            plan = tuple(t for t, _ in sc.events[:-1]) + (begin + lag,)
            r = static_run(sc, plan)
            truth = r["plant"]["angle"][WARM_TICKS:]
            _, settle, ep, good = error_measures(truth, ref, end, begin)
            out.append((seed, lag, begin, ref, r["plant"]["pwm_angle"], truth, settle, ep, good))
    return out


def recovery_fit(data, memory):
    candidates = list(product(
        (12, 15, 18, 20, 23, 26, 30, 34),
        (85, 100, 115, 130, 150, 170, 195),
        (.009, .016, .023, .030, .039, .048) if memory else (0.,),
    ))
    all_tune = []
    th = np.array([r[3] for r in data])
    u = np.array([r[4] for r in data])
    actual_t = np.array([r[6] if r[6] is not None else len(th[j])-r[2] for j,r in enumerate(data)])
    good = np.array([r[8] for r in data]).astype(bool)
    # The cases all share the final maneuver index.
    begin = data[0][2]
    assert all(r[2] == begin for r in data)
    for ii in range(0, len(candidates), 48):
        chunk = candidates[ii:ii+48]
        gains, caps, taus = (np.array(x)[:, None, None] for x in zip(*chunk))
        alpha = 1 - np.exp(-DT / np.where(taus > 0, taus, 1e-100))
        q = np.tile(u[:, 0, :], (len(chunk), 1, 1))
        v = np.zeros_like(q)
        last = np.full((len(chunk),len(data)), -1, dtype=int)
        for k in range(u.shape[1]):
            t = k - WARM_TICKS
            if t >= begin:
                out = np.max(np.abs(q - th[:, t, :]),axis=-1) > ERROR_BAND
                last = np.where(out,t-begin,last)
            req = np.clip(gains * (u[:, k, :] - q),-caps,caps)
            v += alpha * (req - v)
            q += DT*v
        settled = last+1
        predicted = settled<=RECOVERY_TICKS
        mae = np.mean(np.abs(settled-actual_t),axis=1)*DT
        unsafe = np.sum(predicted & ~good,axis=1)
        conservative = np.sum(~predicted & good,axis=1)
        loss = mae + (.040*unsafe + .020*conservative) / len(data)
        for par, l, err, f, g in zip(chunk,loss,mae,unsafe,conservative):
            m = Predictor(*par)
            all_tune.append({"representation":"memory" if memory else "stateless",
                             **asdict(m), "loss":float(l), "recovery_mae_s":float(err),
                             "false_feasible":int(f), "false_infeasible":int(g)})
    best = min(all_tune,key=lambda x:(x["loss"], x["false_feasible"], x["recovery_mae_s"]))
    return Predictor(*(best[k] for k in ("gain","speed","velocity_memory_s"))), all_tune


def assess(model, name, profile, scs):
    rows = []
    for seed, lag, begin, ref, u, truth, real, true_peak, true_pass in scs:
        hat = model.run(u)[0][WARM_TICKS:]
        _, guess, ppeak, ppass = error_measures(hat, ref, len(ref), begin)
        penalize = len(ref) - begin
        delta = abs((penalize if real is None else real)-(penalize if guess is None else guess))
        rows.append({
            "profile":profile,"seed":seed,"lag_ms":lag*10,"model":name,
            "actual_recovery_s":real*DT if real is not None else None,
            "predicted_recovery_s":guess*DT if guess is not None else None,
            "error_s":delta*DT,"false_feasible":int(ppass and not true_pass),
            "false_infeasible":int(not ppass and true_pass),
            "actual_fail":int(not true_pass),"predicted_peak_deg":ppeak,"actual_peak_deg":true_peak,
            "angle_rmse_deg":float(np.sqrt(np.mean((hat[begin:]-truth[begin:])**2))),
        })
    return rows


def run_recovery_refit(out):
    write_json(out/"g1_refit_protocol.json",{
        "fit_seeds":CALIBRATION_SEEDS,"heldout_seeds":CONFIRM_SEEDS,"lags_ms":[x*10 for x in LAGS],
        "criterion":"recovery MAE in seconds + 0.040 * false-feasible share + 0.020 * false-infeasible share",
        "gate":"Same as primary: memory error below 50 ms and 10 ms lower than stateless with no extra "
               "false feasible, nominal and at least one model shift.",
        "relationship":"Follow-up diagnosis using independent scenarios after primary G1 result.",
    })
    cal=cases("nominal",CALIBRATION_SEEDS)
    models={}; tune=[]
    for name in ("stateless","memory"):
        models[name], grid = recovery_fit(cal, name=="memory")
        tune.extend(grid)
    checks=[]
    summary={}
    for profile,seeds in CONFIRM_SEEDS.items():
        bench=cases(profile,seeds)
        summary[profile]={}
        for name,model in models.items():
            a=assess(model,name,profile,bench)
            checks.extend(a)
            summary[profile][name]={
                "n":len(a),"recovery_mae_s":mean(x["error_s"] for x in a),
                "false_feasible":sum(x["false_feasible"] for x in a),
                "actual_failures":sum(x["actual_fail"] for x in a),
                "angle_rmse_deg":mean(x["angle_rmse_deg"] for x in a),
            }
    met = [p for p,b in summary.items()
           if b["memory"]["recovery_mae_s"] < .050
           and b["stateless"]["recovery_mae_s"]-b["memory"]["recovery_mae_s"] >= .010
           and b["memory"]["false_feasible"]<=b["stateless"]["false_feasible"]]
    report={"profiles_met":met, "gate":("nominal" in met and len(met)>1),
            "validation":summary, "model":{k:asdict(v) for k,v in models.items()}}
    write_csv(out/"g1_refit_tuning.csv",tune)
    write_csv(out/"g1_refit_validation.csv",checks)
    write_json(out/"g1_refit_report.json",report)
    return report, models

