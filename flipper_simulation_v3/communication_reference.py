"""Communication reference curves for fixed and threshold policies."""
from __future__ import annotations
from collections import defaultdict
from itertools import product
from statistics import mean, median
import numpy as np

from platform_model import UPDATE_SLOT, steering_trace
from study import (
    FRAME_CAPS, TUNING_SEED, G2_SEED, THRESHOLDS, DWELL, PROFILES,
    for_seed, simple_ticks, static_run, write_csv, write_json,
)

PHASES = (.2, .4, .6, .8)


def periodic(horizon, cap, phase):
    return [round(((i + phase) * horizon / cap) / UPDATE_SLOT) * UPDATE_SLOT
            for i in range(cap)]


def key_metric(items):
    tail_n = max(1, round(.10 * len(items)))
    tail = sorted((x["peak_scored"] for x in items), reverse=True)
    return (-sum(x["passed"] for x in items), mean(tail[:tail_n]),
            mean(x["updates"] for x in items))


def settings():
    return {
        "periodic": tuple((p, None) for p in PHASES),
        "threshold": tuple((thr, 0) for thr in THRESHOLDS),
        "threshold_dwell": tuple(product(THRESHOLDS, DWELL)),
    }


def dispatch(scenario, command, policy, setting, cap):
    if policy == "periodic":
        return periodic(scenario.horizon, cap, setting[0])
    return simple_ticks(command, "threshold", scenario.horizon, val=setting[0], dwell=setting[1], cap=cap)


def run_references(out):
    train = [for_seed(TUNING_SEED+i,i,"comm_cal","nominal") for i in range(12)]
    test = {p:[for_seed(G2_SEED+j*40+i,i,"comm_heldout",p) for i in range(24)]
            for j,p in enumerate(PROFILES)}
    rules = settings()
    write_json(out/"communication_reference_protocol.json", {
        "tuning": [s.seed for s in train],
        "held_out": {p: [s.seed for s in a] for p,a in test.items()},
        "policies": rules,
        "selection": "first maximize task passes, then minimize mean of the worst 10% peaks, "
                     "then minimize applications sent",
        "caps": FRAME_CAPS,
        "scope": "reference curves. The history-aware candidate requires a positive G1 finding.",
    })
    tune, choice = [], {}
    caches = {s.name: {} for s in train}
    commands = {s.name: steering_trace(s) for s in train}
    for cap in FRAME_CAPS:
        for policy, candidates in rules.items():
            sweep = []
            for parameters in candidates:
                subs = []
                for s in train:
                    seq = tuple(dispatch(s, commands[s.name], policy, parameters, cap))
                    cache = caches[s.name]
                    if seq not in cache:
                        cache[seq] = static_run(s, seq)["metrics"]
                    subs.append(cache[seq])
                a,b,c = key_metric(subs)
                tune.append({"cap":cap,"policy":policy,"x":parameters[0],"y":parameters[1],
                             "passed":-a,"tail_peak_deg":b,"mean_updates":c})
                sweep.append(((a,b,c),parameters))
            choice[(cap,policy)] = min(sweep)[1]

    raw, packet_rows = [], []
    for profile, subset in test.items():
        for s in subset:
            cmd = steering_trace(s)
            opts = [(0,"every_change",simple_ticks(cmd,"change",s.horizon)),
                    (0,"zero_update",[])]
            opts += [(cap,pol,dispatch(s,cmd,pol,params,cap)) for (cap,pol),params in choice.items()]
            cache = {}
            for cap,policy,ts in opts:
                seq = tuple(ts)
                if seq not in cache:
                    r = static_run(s,seq)
                    cache[seq] = r["metrics"], r["packets"]
                metrics, packets = cache[seq]
                raw.append({"case":s.name,"profile":profile,"family":s.family,
                            "policy":policy,"cap":cap,**metrics})
                for tick,due,cmd,payload in packets:
                    packet_rows.append({"case":s.name,"policy":policy,"cap":cap,"sent_s":tick*.01,
                                        "arrived_s":due*.01,"offset_deg":cmd,
                                        "application_body":payload.decode("ascii")})
                if cap:
                    assert metrics["updates"] <= cap
    summary = []
    groups = defaultdict(list)
    for r in raw:
        groups[r["profile"],r["policy"],r["cap"]].append(r)
    for (profile,policy,cap),a in sorted(groups.items()):
        passes,tail,updates=key_metric(a)
        summary.append(dict(profile=profile,policy=policy,cap=cap,passed=-passes,n=len(a),
                            tail_peak_deg=tail,mean_updates=updates,
                            mean_application_bytes=mean(x["body_bytes"] for x in a),
                            median_scored_peak_deg=median(x["peak_scored"] for x in a)))
    write_csv(out/"communication_reference_tuning.csv",tune)
    write_csv(out/"communication_reference_trials.csv",raw)
    write_csv(out/"communication_reference_packets.csv",packet_rows)
    write_csv(out/"communication_reference_summary.csv",summary)
    write_json(out/"communication_reference_report.json",{
        "status":"references_completed",
        "frozen_parameters":[{"cap":cap,"policy":name,"parameters":v} for (cap,name),v in choice.items()],
        "scenes": {p:[s.name for s in a] for p,a in test.items()},
        "summary":summary,
    })
    return summary

