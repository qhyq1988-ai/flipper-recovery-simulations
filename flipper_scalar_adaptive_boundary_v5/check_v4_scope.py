"""Replay the earlier periodic-gait example against a relaxed travel limit."""
from pathlib import Path
import argparse
import math
import sys

import numpy as np


def run(v4, output):
    sys.path.insert(0,str(v4))
    from experiment import (fixture, HOLDOUT_FACTORS, servo_bank, TASK_START, WARM_TICKS, DT,
                            pwm, reference, evaluate, plant_outcomes)

    import json
    with (v4/"results"/"diagnostic.json").open(encoding="utf8") as f:
        e=json.load(f)["example"]
    with (v4/"results"/"protocol.json").open(encoding="utf8") as f:
        seeds=json.load(f)["holdout"]
    m=fixture(int(e["seed"]),seeds.index(int(e["seed"])))
    lag=int(e["from_request_ms"])//10
    d=int(e["branch_delay_ms"])//10
    deadline=TASK_START+75
    it=(lag,lag+5)
    target=reference(m.scenario)
    rows=[]
    for profile in ("nominal",e["profile"]):
        for option,ln in zip(("now","wait"),it):
            applied=TASK_START+ln+d
            y=pwm(m,ln,(d,d,d))
            servo=servo_bank(m.seed,HOLDOUT_FACTORS[profile])
            q=y[0].copy()
            velocity=np.zeros_like(q)
            gear=q.copy()
            for i in range(WARM_TICKS+applied):
                q,velocity,gear=servo.step(q,velocity,gear,y[i],i-WARM_TICKS)
            q_hi=q.copy()
            q_lo=q.copy()
            v_hi=velocity.copy()
            v_lo=velocity.copy()
            rates=1-np.exp(-DT/(2*servo.lag))
            # Relax the real position command and allow any demanded velocity within
            # its signed cap. Update in exactly the two substeps of v4's plant.
            for k in range(applied,deadline):
                factor=1-.08*(1+math.sin(k*DT*1.7+servo.phase))
                for _ in range(2):
                    v_hi+=rates*(servo.up*factor-v_hi)
                    v_lo+=rates*(-servo.down*factor-v_lo)
                    q_hi+=DT*v_hi/2
                    q_lo+=DT*v_lo/2
            dist=np.maximum(0.,np.maximum(target[deadline]-q_hi,q_lo-target[deadline]))
            trajectory=plant_outcomes(m.seed,HOLDOUT_FACTORS[profile],y[None,...])
            actual=evaluate(trajectory,target)
            rows.append({
                "profile":profile,"choice":option,"arrival_ms":int((applied-TASK_START)*10),
                "recovery_ms":int(actual[1][0]*10),
                "actual_pass":bool(actual[-1][0]),
                "actual_max_error_at_deadline_deg":float(actual[0][0,deadline]),
                "best_possible_max_error_lb_deg":float(np.max(dist)),
                "travel_limit_proves_deadline_impossible":bool(np.max(dist)>5),
            })
    a={"seed":m.seed,"deadline_ms":750,"cases":rows,
       "interpretation":"Offline necessary test for the exact v4 integrator with arbitrary "
                        "signed bounded velocity requests and known state at packet arrival."}
    output.write_text(json.dumps(a,indent=2)+"\n",encoding="utf8")
    print(a)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--v4",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"results"/"v4_scope_check.json")
    a=p.parse_args()
    run(a.v4.resolve(),a.out)


