"""Predeclared Flipper v3 simulation gates and experimental exports."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import argparse
import csv
import json
from itertools import product
from pathlib import Path
from statistics import mean

import numpy as np

from platform_model import (
    DT, UPDATE_SLOT, WARM_TICKS, CHANNELS, ERROR_BAND, RECOVERY_TICKS,
    TRANSIENT_LIMIT, SOURCE_REVISION, Scenario, replay, steering_trace, delay_trace,
    on_board_offsets, physical_trace, score, reference, turn_body, map_turn_input,
)

ROOT = Path(__file__).resolve().parent
FAMILIES = ("sustained", "alternating", "brief", "ramp")
PROFILES = ("nominal", "loaded", "asymmetric")
G0_SEED = 36110
G1_SEED = 36140
TUNING_SEED = 36200
G2_SEED = 36300
FRAME_CAPS = (2, 4, 8, 16)
THRESHOLDS = (1, 3, 5, 7, 9)
DWELL = (0, 10, 25, 40)
G0_CASES = 16


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, data, names=None):
    if names is None:
        names = list(data[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(data)


def for_seed(seed, index, prefix, profile):
    r = np.random.default_rng(seed)
    family = FAMILIES[index % 4]
    sign = [-1, 1][int(r.integers(0, 2))]
    c1 = sign * int(r.integers(7, 11))
    c2 = -sign * int(r.integers(7, 11))
    c3 = sign * int(r.integers(6, 11))
    if family == "sustained":
        ev = [(115, c1), (455, int(sign * r.integers(0, 4)))]
    elif family == "alternating":
        ev = [(115, c1), (270, c2), (475, c3), (630, 0)]
    elif family == "brief":
        off_length = int(r.choice([20, 35, 45, 55]))
        ev = [(115, c1), (115 + off_length, 0), (265, c2),
              (415, c3), (415 + off_length, c2), (575, c3)]
    else:
        ev = [(115 + 20 * j, int(round(c1 * j / 5))) for j in range(1, 6)]
        ev += [(295 + 20 * j, int(round(c1 + (c2 - c1) * j / 8))) for j in range(1, 9)]
        ev += [(560, c3)]
    phase = int(r.integers(0, 360))
    delay = {"nominal": (8, 3), "loaded": (10, 4), "asymmetric": (13, 5)}[profile]
    return Scenario(f"{prefix}_{seed}", family, seed, phase, tuple(ev), profile=profile,
                    delay=delay[0], jitter=delay[1])


def commit_schedule(scenario, ticks, command, lags):
    arrivals = []
    last_due = -1
    for t in ticks:
        assert t % UPDATE_SLOT == 0
        due = max(t + int(lags[t]), last_due + 1)
        payload = turn_body(int(command[t]))
        arrivals.append((t, due, int(command[t]), payload))
        last_due = due
    return arrivals


def simple_ticks(command, policy, horizon, val=0, dwell=0, cap=1_000, phase=0):
    result = []
    held, stable_since = 0, 0
    for t in range(0, horizon, UPDATE_SLOT):
        current = int(command[t])
        if t and current != command[t - 1]:
            stable_since = t
        if len(result) == cap:
            break
        if policy == "none":
            continue
        if policy == "change":
            send = current != held
        elif policy == "periodic":
            send = (t - phase) % val == 0
        elif policy == "threshold":
            send = abs(current - held) >= val and t - stable_since >= dwell
        else:
            raise ValueError(policy)
        if send:
            result.append(t)
            held = current
    return result


def static_run(s, ticks=None, current=False):
    command = steering_trace(s)
    if current:
        arrived = {k: c for k, c in s.events}
        packets = []
    else:
        packets = commit_schedule(s, ticks or [], command, delay_trace(s))
        arrived = {due: value for _, due, value, _ in packets}
    held = on_board_offsets(s.horizon, arrived)
    plant = physical_trace(s, held)
    angles = plant["angle"][WARM_TICKS:]
    metric, error, windows = score(s, angles)
    return result_record(s, packets, held, plant, metric, error, windows)


def result_record(scenario, packets, held, plant, metric, error, windows):
    return {
        "metrics": {
            "updates": len(packets),
            "body_bytes": sum(len(body) for _, _, _, body in packets),
            **metric,
        },
        "held": held,
        "plant": plant,
        "error": error,
        "windows": windows,
        "packets": packets,
    }


def trace_csv(path, scenario, rr):
    plant = rr["plant"]
    angles = plant["angle"][WARM_TICKS:]
    desired = reference(scenario)
    pwm = plant["pwm_tick"][WARM_TICKS:]
    remote = steering_trace(scenario)
    arrivals = {du: (t, c, len(b)) for t, du, c, b in rr["packets"]}
    rows = []
    for k in range(scenario.horizon + 1):
        tx = arrivals.get(k)
        row = {"t_s": k * DT, "remote_deg": int(remote[k]), "applied_offset_deg": int(rr["held"][k]),
               "arrived_sent_s": tx[0] * DT if tx else None, "peak_abs_deg": float(rr["error"][k])}
        for j in range(CHANNELS):
            row[f"joint{j}_deg"] = float(angles[k, j])
            row[f"target{j}_deg"] = float(desired[k, j])
            row[f"pwm{j}_count"] = int(pwm[k, j])
        rows.append(row)
    write_csv(path, rows)


def initial_manifest():
    return {
        "source_commit": SOURCE_REVISION,
        "primary": {"band_deg": ERROR_BAND, "deadline_s": RECOVERY_TICKS * DT,
                    "whole_trial_peak_deg": TRANSIENT_LIMIT},
        "sensitivity": {"bands_deg": [3, 5, 7], "deadlines_s": [.50, .75, 1.]},
        "g0_seeds": list(range(G0_SEED, G0_SEED + G0_CASES)),
        "identification_seeds": {"fit": list(range(G1_SEED, G1_SEED + 9)),
                                  "nominal_validation": list(range(G1_SEED + 10, G1_SEED + 16)),
                                  "shift_validation": list(range(G1_SEED + 20, G1_SEED + 32))},
        "g2_tuning_seeds": list(range(TUNING_SEED, TUNING_SEED + 12)),
        "g2_holdout_seeds": {p: list(range(G2_SEED + i*40, G2_SEED+i*40+24))
                             for i, p in enumerate(PROFILES)},
        "budget_caps": FRAME_CAPS,
        "communication_slot_s": UPDATE_SLOT * DT,
        "clock_step_s": DT,
        "assumptions": {
            "board_sweep_s": .02, "channel_writes": "2 groups at 0 and 10 ms",
            "transport": "synthetic jitter, FIFO, no drops, no sender observation of delivery or q",
            "wire_count": "off=<integer> application form bytes, excluding HTTP/TCP/Wi-Fi headers",
            "plant_provenance": "synthetic closed-loop angle response; ADC/PWM timing inferred by a scenario assumption",
        },
    }


def g0(out):
    dev = [for_seed(G0_SEED + i, i, "gate", PROFILES[(i//4) % 3]) for i in range(G0_CASES)]
    summary, sweep, thresholds = [], [], []
    prediction_metadata = {}
    for s in dev:
        command = steering_trace(s)
        basics = {
            "zero_update": static_run(s, []),
            "every_change": static_run(s, simple_ticks(command, "change", s.horizon)),
            "ideal_arrival": static_run(s, current=True),
            "periodic_1s": static_run(s, simple_ticks(command, "periodic", s.horizon, val=100, phase=30)),
        }
        for label, rr in basics.items():
            summary.append({"scene": s.name, "profile": s.profile, "family": s.family,
                            "policy": label, **rr["metrics"]})
        for label in ("zero_update", "every_change", "periodic_1s"):
            for (band, ticks) in product((3, 5, 7), (50, 75, 100)):
                m, _, __ = score(s, basics[label]["plant"]["angle"][WARM_TICKS:], band, ticks)
                sweep.append({"scene": s.name, "profile": s.profile, "family": s.family,
                              "policy": label, "band": band, "deadline_s": ticks * DT,
                              "passed": m["passed"], "peak_scored": m["peak_scored"]})
        cache = {}
        for threshold, dwell in product(THRESHOLDS, DWELL):
            ts = tuple(simple_ticks(command, "threshold", s.horizon, val=threshold, dwell=dwell))
            if ts not in cache:
                cache[ts] = static_run(s, ts)["metrics"]
            thresholds.append({"scene": s.name, "threshold": threshold, "dwell_s": dwell * DT, **cache[ts]})
        prediction_metadata[s.name] = {"scenario": asdict(s), "change_run": basics["every_change"]}

    rows = defaultdict(list)
    for line in thresholds:
        rows[(line["threshold"], line["dwell_s"])].append(line)
    best = min(rows, key=lambda k: (-sum(x["passed"] for x in rows[k]),
                                   mean(x["updates"] for x in rows[k]),
                                   mean(x["peak_scored"] for x in rows[k])))
    best_scenes = {x["scene"]: x for x in rows[best]}
    witness = []
    # Search retrospectively for sample-level headroom, with all sends at command changes.
    # This search cannot certify generally feasible online decisions.
    for s in dev:
        if s.family not in ("brief", "alternating"):
            continue
        chosen = simple_ticks(steering_trace(s), "change", s.horizon)
        fresh = prediction_metadata[s.name]["change_run"]
        if not fresh["metrics"]["passed"]:
            continue
        for k in list(chosen):
            candidate = [j for j in chosen if j != k]
            rr = static_run(s, candidate)
            if rr["metrics"]["passed"]:
                chosen = candidate
        result = static_run(s, chosen)
        base = best_scenes[s.name]
        if result["metrics"]["passed"] and (not base["passed"] or len(chosen) < base["updates"]):
            witness.append({"scene": s.name, "family": s.family, "threshold": base,
                            "offline_ticks": chosen, "offline_metrics": result["metrics"]})
            if len(witness) == 1:
                for p, rr in (("fresh", fresh), ("retrospective", result),
                              ("threshold", static_run(s, simple_ticks(steering_trace(s), "threshold",
                                  s.horizon, val=best[0], dwell=round(best[1]/DT))))):
                    trace_csv(out / f"g0_witness_{p}.csv", s, rr)
    totals = {p: sum(x["passed"] for x in summary if x["policy"] == p)
              for p in ("zero_update", "every_change", "ideal_arrival", "periodic_1s")}
    informative = [x["scene"] for x in summary if x["policy"] == "every_change" and x["passed"]
                   and not any(y["scene"] == x["scene"] and y["policy"] == "periodic_1s" and y["passed"]
                               for y in summary)]
    gate = bool(informative and witness)
    report = {"gate": gate, "scenes": len(dev), "totals": totals, "informative": informative,
              "best_threshold": list(best),
              "best_threshold_pass": sum(x["passed"] for x in rows[best]),
              "witness": witness, "scenarios": [asdict(s) for s in dev]}
    write_csv(out/"g0_summary.csv", summary)
    write_csv(out/"g0_sensitivity.csv", sweep)
    write_csv(out/"g0_thresholds.csv", thresholds)
    write_json(out/"g0_report.json", report)
    return report


def run_later(out):
    from identification_study import run_identification
    from recovery_refit import run_recovery_refit
    from communication_reference import run_references

    primary, _ = run_identification(out)
    print("G1 primary", primary["gate"], flush=True)
    refit, _ = run_recovery_refit(out)
    print("G1 recovery-aligned check", refit["gate"], flush=True)
    run_references(out)
    g1_pass = primary["gate"] or refit["gate"]
    write_json(out / "stage_status.json", {
        "task_screen": "passed",
        "identification": "passed" if g1_pass else "failed",
        "communication_reference": "complete",
        "history_aware_trigger": "requires implementation" if g1_pass else "stopped at approved identification gate",
    })
    print("Communication reference curves complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    category = parser.add_mutually_exclusive_group()
    category.add_argument("--g0", action="store_true")
    category.add_argument("--continue-g1", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.continue_g1:
        report = json.loads((args.out / "g0_report.json").read_text(encoding="utf-8"))
    else:
        write_json(args.out / "manifest.json", initial_manifest())
        report = g0(args.out)
    print("G0", report["gate"], report["totals"], f'{len(report["witness"])} headroom witnesses', flush=True)
    if args.g0 or not report["gate"]:
        return
    run_later(args.out)


if __name__ == "__main__":
    main()


