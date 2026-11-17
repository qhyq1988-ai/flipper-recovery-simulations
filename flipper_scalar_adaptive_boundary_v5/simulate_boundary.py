"""Independent RK4 tests of the deadline formula and adaptation examples."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import random

from boundary import (
    Interval, Problem, completion, continuous_boundary, latest_slot, speed_update,
    gain_update, worst,
)

ROOT=Path(__file__).resolve().parent
BASE=Problem(step=18.,band=5.,deadline=.75,period=.05,e0=Interval(16,20),
             gain=Interval(18,28),forward=Interval(65,155),reverse=Interval(72,160),
             delay=Interval(.05,.11))
CALIBRATION_V=112.
CALIBRATION_K=22.
SENSOR_ERROR=.075
INTEGRATION_STEP=.001


def csv_rows(out, rows):
    with out.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def json_result(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")


def numerical(send,delay,e0,gain,forward,reverse,sign,problem,delta=INTEGRATION_STEP,
              horizon=1.10, keep=False):
    """Integrate the dimensional q equation, with no analytic flow or band-time formula."""
    deadline=problem.deadline
    arrival=send+delay
    old=-9.*sign
    new=old+sign*problem.step
    q=new-sign*e0
    neg,pos=(reverse,forward) if sign>0 else (forward,reverse)
    current=0.
    first_band=None
    error_at_deadline=None
    points=[(0.,q,abs(q-new))] if keep else None
    jumps=sorted({a for a in (deadline,arrival,horizon) if 0.<a<=horizon})
    for end in jumps:
        target=old if current<arrival-1.e-12 else new
        while end-current>1e-13:
            h=min(delta,end-current)
            k1=max(-neg,min(pos,gain*(target-q)))
            k2=max(-neg,min(pos,gain*(target-q-.5*h*k1)))
            k3=max(-neg,min(pos,gain*(target-q-.5*h*k2)))
            k4=max(-neg,min(pos,gain*(target-q-h*k3)))
            qn=q+h*(k1+2*k2+2*k3+k4)/6
            eold,enew=abs(q-new),abs(qn-new)
            if first_band is None and eold>problem.band>=enew:
                first_band=current+h*(eold-problem.band)/(eold-enew)
            current+=h
            q=qn
            if keep:
                points.append((current,q,enew))
        if abs(end-deadline)<1.e-12:
            error_at_deadline=abs(q-new)
    return first_band,error_at_deadline,points


def requirement(p,send):
    """Smallest admissible lower forward cap making this send slot robust."""
    if worst(p,send)<=p.deadline:
        return p.forward.lo
    def cost(v):
        return worst(replace(p,forward=Interval(v,p.forward.hi)),send)
    lo,hi=p.forward.lo,p.forward.hi
    if cost(hi)>p.deadline:
        return None
    for _ in range(65):
        m=(lo+hi)/2
        if cost(m)<=p.deadline:
            hi=m
        else:
            lo=m
    return hi


def describe(name,p):
    slot=latest_slot(p)
    return {
        "mode":name, "problem":asdict(p), "continuous_s":continuous_boundary(p),
        "last_slot":slot, "last_send_s":None if slot is None else slot*p.period,
    }


def observe_motion(initial,interval,v,k,noise_a,noise_b):
    # Generate the probe by a separate RK4 integration.
    remaining=initial
    t=0.
    while t<interval-1.e-12:
        h=min(.0005,interval-t)
        a=-min(v,k*remaining)
        b=-min(v,k*(remaining+h*a/2))
        c=-min(v,k*(remaining+h*b/2))
        d=-min(v,k*(remaining+h*c))
        remaining+=h*(a+2*b+2*c+d)/6
        t+=h
    return initial+noise_a,remaining+noise_b


def calibration(out):
    p=BASE
    stages=[describe("command_history",p)]
    samples=[]
    for j,(start,duration,err_a,err_b) in enumerate(((20.,.06,.05,-.02),(20.,.06,-.045,.04),
                                                    (6.3,.02,.03,-.01)),1):
        a,b=observe_motion(start,duration,CALIBRATION_V,CALIBRATION_K,err_a,err_b)
        interval,state=speed_update(p.forward,p.gain.lo,a,b,duration,SENSOR_ERROR)
        samples.append({
            "sample":j,"duration_s":duration,"observed_gap_start_deg":a,"observed_gap_end_deg":b,
            "certification_threshold_deg":p.forward.hi/p.gain.lo,
            "reading_error_deg":SENSOR_ERROR,"update":state,
            "prior_min_deg_s":p.forward.lo,"prior_max_deg_s":p.forward.hi,
            "posterior_min_deg_s":interval.lo,"posterior_max_deg_s":interval.hi,
        })
        assert interval.lo<=CALIBRATION_V<=interval.hi
        p=replace(p,forward=interval)
        stages.append(describe(f"after_sample_{j}",p))
    csv_rows(out/"calibration_samples.csv",samples)
    v_at_slot=requirement(BASE,.50)
    min_duration=4*SENSOR_ERROR/(CALIBRATION_V-v_at_slot)
    max_duration=(20-BASE.forward.hi/BASE.gain.lo-2*SENSOR_ERROR)/BASE.forward.hi
    assert min_duration<=.06<=max_duration
    read_a,read_b=observe_motion(2.5,.06,CALIBRATION_V,CALIBRATION_K,.006,-.010)
    gain,state=gain_update(p.gain,p.forward.lo,read_a,read_b,.06,.025)
    assert gain.lo<=CALIBRATION_K<=gain.hi and state=="updated"
    # A separate near-equilibrium probe only demonstrates the gain equation.
    stats={
        "stages":stages,"examples":samples,"forward_speed_required_at_500_ms":v_at_slot,
        "probe_duration_s_for_all_sensor_errors":{"min_s":min_duration,"max_s":max_duration},
        "illustrative_true_probe":{"v_deg_s":CALIBRATION_V,"gain_inv_s":CALIBRATION_K,
                                   "initial_gap_deg":20.,"max_angular_error_deg":SENSOR_ERROR},
        "near_equilibrium_gain_probe":{"description":"separate optional probe, not used in main bound",
                                       "gain_interval":asdict(gain),"error_bound_deg":.025,"decision":state},
    }
    return p,stats


def verify_grids(out, adapted):
    rng=random.Random(16331)
    rows=[]; worlds_total=0; disagreement=0; false_safe=0; max_abs=0.
    for name,p in (("prior",BASE),("adaptive",adapted)):
        params=(p.e0,p.gain,p.forward,p.reverse,p.delay)
        corner=list(product(*[(u.lo,u.hi) for u in params]))
        interior=[tuple(rng.uniform(u.lo,u.hi) for u in params) for _ in range(28)]
        for sign in (1,-1):
            for slot in range(16):
                send=slot*p.period
                values=[]
                for e,k,f,r,d in corner+interior:
                    n,err,_=numerical(send,d,e,k,f,r,sign,p)
                    a=completion(send,d,e,k,f,r,p.step,p.band)
                    if n is not None:
                        max_abs=max(max_abs,abs(a-n))
                    else:
                        assert a>1.10-1e-8
                    observed=err<=p.band+2.e-7
                    theoretical=a<=p.deadline+1.e-12
                    disagreement+=observed!=theoretical
                    false_safe+=observed==False and worst(p,send)<=p.deadline+1.e-12
                    if n is not None:
                        values.append(n)
                    worlds_total+=1
                # The worst corner always has recovery within numerical horizon for the example.
                nn,err,_=numerical(send,p.delay.hi,p.e0.hi,p.gain.lo,p.forward.lo,p.reverse.lo,sign,p)
                assert nn is not None
                theoretical=worst(p,send)
                rows.append({
                    "mode":name,"sign":sign,"slot":slot,"send_s":send,
                    "analytic_worst_s":theoretical,"rk4_corner_s":nn,
                    "largest_sampled_rk4_s":max(values),
                    "worst_angle_error_at_deadline_deg":err,
                    "bound_feasible":int(theoretical<=p.deadline+1.e-12),
                })
                assert abs(nn-theoretical)<5e-6
                assert abs(max(values)-nn)<5e-6
    assert not false_safe and not disagreement and max_abs<5e-6
    csv_rows(out/"grid_verification.csv",rows)
    return {"case_paths":worlds_total,"false_safe_count":false_safe,"classification_disagreement":disagreement,
            "largest_timing_difference_s":max_abs,"integration_step_s":INTEGRATION_STEP}


def stress_examples(out, adapted):
    drift=35.
    widened=Interval(max(BASE.forward.lo,adapted.forward.lo-drift),
                     min(BASE.forward.hi,adapted.forward.hi+drift))
    drift_aware=replace(adapted,forward=widened)
    examples=[
        ("prior_last",.40,BASE,BASE.forward.lo),
        ("prior_next",.45,BASE,BASE.forward.lo),
        ("adaptive_last",.50,adapted,adapted.forward.lo),
        ("adaptive_next",.55,adapted,adapted.forward.lo),
        ("stale_bound_after_load_change",.50,adapted,80.),
        ("drift_widened_last",latest_slot(drift_aware)*drift_aware.period,drift_aware,widened.lo),
    ]
    summaries=[]; traces=[]
    for name,s,p,true_speed in examples:
        cr,err,line=numerical(s,p.delay.hi,p.e0.hi,p.gain.lo,true_speed,p.reverse.lo,1,p,keep=True)
        summaries.append({
            "case":name,"send_s":s,"arrival_s":s+p.delay.hi,"true_forward_cap_deg_s":true_speed,
            "used_forward_interval":asdict(p.forward),"predicted_worst_recovery_s":worst(p,s),
            "numerical_recovery_s":cr,"deadline_error_deg":err,
            "passed":int(err<=p.band+1.e-7),
        })
        traces.extend({"case":name,"t_s":t,"angle_deg":q,"absolute_error_deg":e} for t,q,e in line[::4])
    assert [r["passed"] for r in summaries]==[1,0,1,0,0,1]
    csv_rows(out/"verification_traces.csv",traces)
    return {"cases":summaries,"drift_bound_deg_s":drift,"drift_aware_margin":describe("drift_widened",drift_aware)}


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    adapted,identification=calibration(out)
    checked=verify_grids(out,adapted)
    ranges=stress_examples(out,adapted)
    stats={"adaptive":identification,"grid":checked,"stress":ranges}
    json_result(out/"report.json",stats)
    print(json.dumps({
        "stages":[(s["mode"],s["continuous_s"],s["last_send_s"]) for s in identification["stages"]],
        "speed_required_for_500ms":identification["forward_speed_required_at_500_ms"],
        "verifications":checked,
        "cases":[(x["case"],x["numerical_recovery_s"],x["passed"]) for x in ranges["cases"]],
    },indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=ROOT/"results")
    run(p.parse_args().out)


