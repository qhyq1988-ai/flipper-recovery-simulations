"""Summarize paired decisions and export representative recovery traces."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import csv
import json
from statistics import mean
import numpy as np

from experiment import (
    DT, WARM_TICKS, TASK_START, END, PUBLIC_DELAY, HOLDOUT_FACTORS, Predictor,
    fixture, pwm, plant_outcomes, evaluate, reference, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def analyze(out):
    decisions=read(out/"paired_decisions.csv")
    truth=read(out/"paired_outcomes.csv")
    protocol=load(out/"protocol.json")
    report=load(out/"report.json")
    stateless,memory=report["selected_stateless"],report["selected_with_memory"]
    test=[r for r in decisions if r["split"]=="heldout"]
    groups={}
    for transport in ("fixed","jitter"):
        a={r["key"]:r for r in test if r["transport"]==transport and r["model"]==stateless}
        b={r["key"]:r for r in test if r["transport"]==transport and r["model"]==memory}
        assert a.keys()==b.keys()
        d={}
        for status in ("raw_decision","calibrated_decision"):
            d[status]={
                "total":len(a),
                "changed":sum(a[k][status]!=b[k][status] for k in a),
                "table":{f"{x} -> {y}":n for (x,y),n in sorted(Counter(
                     (a[k][status],b[k][status]) for k in a).items())},
            }
        groups[transport]=d

    rows=[r for r in truth if r["transport"]=="fixed" and r["split"]=="heldout"]
    nom={(r["seed"],r["from_request_ms"]):r for r in rows if r["profile"]=="nominal"}
    conflicts=[]
    candidate=[]
    for profile in HOLDOUT_FACTORS:
        if profile=="nominal":
            continue
        xs=[r for r in rows if r["profile"]==profile]
        c=[r for r in xs if nom[r["seed"],r["from_request_ms"]]["outcome"]=="both_ok"
           and r["outcome"]=="urgent"]
        candidate.extend(c)
        conflicts.append({
            "profile":profile,"nominal_wait_ok_shift_now_only":len(c),
            "nominal_wait_ok_shift_neither":sum(nom[r["seed"],r["from_request_ms"]]["outcome"]=="both_ok"
                  and r["outcome"]=="neither" for r in xs),
            "pair_coordinates":len(xs),
        })
    chosen=min(candidate,key=lambda r:(r["seed"],r["from_request_ms"],r["profile"])) if candidate else None
    example=None
    if chosen:
        seed=int(chosen["seed"])
        m=fixture(seed,protocol["holdout"].index(seed))
        desired=reference(m.scenario)
        lag=int(chosen["from_request_ms"])//10
        u=np.array([pwm(m,t,(PUBLIC_DELAY,)*3) for t in (lag,lag+5)])
        by_profile=("nominal",chosen["profile"])
        display=[]
        for p in by_profile:
            a=evaluate(plant_outcomes(seed,HOLDOUT_FACTORS[p],u),desired)
            for i,option in enumerate(("now","wait")):
                for k in range(TASK_START,END+1):
                    display.append({"profile":p,"choice":option,
                                    "ms_from_request":(k-TASK_START)*10,
                                    "maximum_error_deg":float(a[0][i,k])})
        write_csv(out/"illustrated_pair.csv",display)
        pred=[r for r in test if r["key"]==chosen["key"] and r["model"] in (stateless,memory)]
        example={**chosen,"same_input_nominal":nom[chosen["seed"],chosen["from_request_ms"]],
                 "predictions":{r["model"]:r["calibrated_decision"] for r in pred}}
    result={
        "selected_pairings":groups,
        "hidden_plant_conflicts":conflicts,
        "example":example,
    }
    write_json(out/"diagnostic.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=ROOT/"results")
    z=analyze(p.parse_args().out)
    print(json.dumps(z["selected_pairings"],ensure_ascii=False))

