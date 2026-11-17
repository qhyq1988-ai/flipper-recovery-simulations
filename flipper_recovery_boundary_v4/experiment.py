"""Paired Flipper recovery experiments with independent role splits."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import argparse
import csv
from itertools import product
import json
from pathlib import Path
from statistics import mean

import numpy as np

from platform_model import (
    CHANNELS, DT, WARM_TICKS, OFFSET_LIMIT, ERROR_BAND, RECOVERY_TICKS,
    SOURCE_REVISION, TRANSIENT_LIMIT, Scenario, VirtualServo, replay,
    on_board_offsets, board_trajectory, reference,
)

ROOT = Path(__file__).resolve().parent
TASK_START = 230
END = 409
SEND_LAGS = (45, 50, 55, 60, 65)
WAIT = 5
PUBLIC_DELAY = 8
CENSOR = END + 1 - TASK_START
PROFILES = ("nominal", "load", "direction", "lag", "combined")
FIT = tuple(range(45100,45132))
CAL = tuple(range(45200,45224))
TEST = tuple(range(45400,45432))
MODEL_SPEC = {
    "sym_simple": (False,False),
    "sym_memory": (False,True),
    "asym_simple": (True,False),
    "asym_memory": (True,True),
}
GATE_PAIRS = ("urgent", "both_ok")
RISK_CAP = .05


@dataclass(frozen=True)
class Factors:
    scale: float = 1.
    negative: float = 1.
    lag: float = 1.


NOMINAL = Factors()
HOLDOUT_FACTORS = {
    "nominal": NOMINAL,
    "load": Factors(scale=.76),
    "direction": Factors(negative=.67),
    "lag": Factors(lag=1.85),
    "combined": Factors(scale=.85,negative=.76,lag=1.45),
}
FIT_FACTORS = (NOMINAL, Factors(scale=.88), Factors(negative=.82), Factors(lag=1.32))
CAL_FACTORS = (NOMINAL, Factors(scale=.83), Factors(negative=.76), Factors(lag=1.55),
               Factors(scale=.92,negative=.87,lag=1.23))


@dataclass(frozen=True)
class Maneuver:
    seed: int
    phase: int
    sign: int
    scenario: Scenario


@dataclass(frozen=True)
class Outcome:
    feasible: bool
    peak: float
    peak_all: float
    recovery: int | None


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def write_csv(path, rows):
    assert rows, path
    with path.open("w", newline="", encoding="utf8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fixture(seed, i):
    rng = np.random.default_rng(seed ^ 61301)
    sign = 1 if i % 2 == 0 else -1
    phase = (i//2 % 4)*90 + int(rng.integers(-12,13))
    earlier = sign*int(rng.integers(4,10))
    last = -sign*int(rng.integers(6,11))
    value = sign*int(rng.integers(4,11))
    if i//8 % 4 == 3:
        last = -sign*int(rng.integers(1,5))
        value = sign*int(rng.integers(2,6))
    prefix_end = int(rng.choice([120,155,190,210]))
    s = Scenario(f"episode_{seed}", "paired_recovery", seed, phase,
                 ((60,earlier),(prefix_end,last),(TASK_START,value)), horizon=END)
    return Maneuver(seed,phase,sign,s)


def jitter_seed(seed):
    rng = np.random.default_rng(seed ^ 99137)
    return tuple(int(x) for x in rng.integers(5,12, size=3))


def applied(m, lag, delay):
    e = m.scenario.events
    result = {t+delay[i]:c for i,(t,c) in enumerate(e[:2])}
    result[TASK_START + lag + delay[2]] = e[2][1]
    return on_board_offsets(END, result)


def pwm(m, lag, delay):
    return board_trajectory(replay(m.phase,END),applied(m,lag,delay))[0]


def servo_bank(seed, factors):
    v = VirtualServo.create(seed,"nominal")
    v.up *= factors.scale
    v.down *= factors.scale * factors.negative
    v.lag *= factors.lag
    return v


def plant_outcomes(seed, factors, targets):
    """Evaluate all counterfactual schedules on the same plant with a common start."""
    v = servo_bank(seed,factors)
    q = targets[:,0,:].copy()
    speed = np.zeros_like(q)
    play = q.copy()
    log = np.empty_like(targets, dtype=float)
    for i in range(targets.shape[1]):
        log[:,i,:] = q
        q,speed,play = v.step(q,speed,play,targets[:,i,:],i-WARM_TICKS)
    return log


def evaluate(traces, desired):
    e = np.max(np.abs(traces[...,WARM_TICKS:,:] - desired),axis=-1)
    post = e[...,TASK_START:]
    late = post>ERROR_BAND
    idx = np.where(late,np.arange(post.shape[-1])+1,0)
    time = np.max(idx,axis=-1)
    peak = np.max(post[...,RECOVERY_TICKS:],axis=-1)
    all_peak = np.max(e,axis=-1)
    feasible = (peak<=ERROR_BAND) & (all_peak<=TRANSIENT_LIMIT)
    return e,time,peak,all_peak,feasible


def one_outcome(outcomes,i):
    _, time, peak, all_peak, feasible = outcomes
    rec = int(time[i])
    return Outcome(bool(feasible[i]),float(peak[i]),float(all_peak[i]),None if rec==CENSOR else rec)


def label(a,b):
    if a and not b:
        return "urgent"
    if a and b:
        return "both_ok"
    if b:
        return "wait_only"
    return "neither"


@dataclass(frozen=True)
class Predictor:
    gain: float
    up: float
    down: float
    memory_s: float

    def simulate(self, target):
        q = target[:,0,:].copy()
        v = np.zeros_like(q)
        trajectory = np.empty_like(target, dtype=float)
        alpha = 1 - np.exp(-DT / self.memory_s) if self.memory_s else 1.
        for k in range(target.shape[1]):
            trajectory[:,k,:]=q
            requested = np.clip(self.gain*(target[:,k,:]-q),-self.down,self.up)
            v += alpha*(requested-v)
            q += DT*v
        return trajectory


def library(directional, memory):
    ratio = (.65,.78,.89,1.,1.12) if directional else (1.,)
    tau = (.010,.020,.030,.045) if memory else (0.,)
    return [Predictor(k,s,s*r,h) for k,s,r,h in product((16,20,24,28,32),
                                                        (95,115,135,155,180),ratio,tau)]


def fitting_set():
    batches,refs,truth = [],[],[]
    for i,seed in enumerate(FIT):
        m = fixture(seed,i)
        lag = (0,15,30,45,50,55,60,65,70)
        tar = np.array([pwm(m,t,(PUBLIC_DELAY,)*3) for t in lag])
        observed = plant_outcomes(seed,FIT_FACTORS[((i//2)%4 + i//8)%4],tar)
        # Offline sensor quantization. The evaluated recovery status uses physical angles.
        sensor = np.random.default_rng(seed ^ 70717)
        observed = observed + sensor.normal(0.,.05,observed.shape)
        batches.extend(tar)
        refs.extend([reference(m.scenario)]*len(lag))
        truth.extend(np.round(observed*10)/10)
    return np.array(batches),np.array(refs),np.array(truth)


def fit_models(out):
    u, desired, angles = fitting_set()
    measured = evaluate(angles,desired)
    actual_r, actual_pass = measured[1],measured[-1]
    history = []
    chosen = {}
    # All four models consume the same applied PWM/quantized angle training sequences.
    # Batches share shape but have independent episodes; the simulation starts at the public PWM state.
    for name,(directional,memory) in MODEL_SPEC.items():
        models = library(directional,memory)
        fit_rows = []
        for start in range(0,len(models),36):
            chunk = models[start:start+36]
            mult = lambda attr: np.array([getattr(m,attr) for m in chunk])[:,None,None]
            g,up,dn,tau = (mult(x) for x in ("gain","up","down","memory_s"))
            alpha = 1-np.exp(-DT/np.where(tau>0,tau,1e-50))
            q = np.repeat(u[None,:,0,:],len(chunk),axis=0)
            v = np.zeros_like(q)
            last = np.zeros(q.shape[:2], dtype=int)
            mp = np.zeros_like(last,dtype=float)
            total = np.zeros_like(last,dtype=float)
            square = np.zeros_like(last,dtype=float)
            for k in range(u.shape[1]):
                taskk = k-WARM_TICKS
                if taskk>=TASK_START:
                    abs_err = np.max(np.abs(q-desired[:,taskk,:]),axis=-1)
                    last = np.where(abs_err>ERROR_BAND,taskk-TASK_START+1,last)
                    if taskk>=TASK_START+RECOVERY_TICKS:
                        mp = np.maximum(mp,abs_err)
                    square += np.mean((q-angles[:,k,:])**2,axis=-1)
                if taskk>=0:
                    total = np.maximum(total,np.max(np.abs(q-desired[:,taskk,:]),axis=-1))
                r = np.clip(g*(u[:,k,:]-q),-dn,up)
                v += alpha*(r-v)
                q += DT*v
            pred = (mp<=ERROR_BAND) & (total<=TRANSIENT_LIMIT)
            mae = np.mean(np.abs(last-actual_r),axis=1)*DT
            err_feas = np.mean(pred & ~actual_pass,axis=1)
            err_infeas = np.mean(~pred & actual_pass,axis=1)
            rmse = np.sqrt(np.mean(square,axis=1)/CENSOR)
            losses = mae + .050*err_feas + .025*err_infeas + .002*rmse
            for model,a,b,c,d,e in zip(chunk,mae,err_feas,err_infeas,rmse,losses):
                fit_rows.append({"model":name,**asdict(model),"mae_s":float(a),"false_feasible_share":float(b),
                                 "false_infeasible_share":float(c),"angle_rmse_deg":float(d),"loss":float(e)})
        picked = min(fit_rows,key=lambda x:(x["loss"],x["false_feasible_share"],x["mae_s"]))
        chosen[name] = Predictor(*(picked[k] for k in ("gain","up","down","memory_s")))
        history.extend(fit_rows)
    write_csv(out/"fit_sweep.csv",history)
    write_json(out/"fitted_models.json",{k:asdict(x) for k,x in chosen.items()})
    return chosen



def records_for_episode(i,seed,split,models):
    m = fixture(seed,i)
    desired = reference(m.scenario)
    lags = (*SEND_LAGS, SEND_LAGS[-1]+WAIT)
    nominal_delay = (PUBLIC_DELAY,)*3
    public_targets = np.array([pwm(m,t,nominal_delay) for t in lags])
    predicted = {name:evaluate(model.simulate(public_targets),desired)
                 for name,model in models.items()}
    if split=="calibration":
        profiles = {f"cal_{j}":f for j,f in enumerate(CAL_FACTORS)}
        transports = ("fixed","jitter")
    else:
        profiles = HOLDOUT_FACTORS
        transports = ("fixed","jitter")
    labels, output = [], []
    for profile,factor in profiles.items():
        for transport in transports:
            delay = nominal_delay if transport=="fixed" else jitter_seed(seed)
            actual_targets = (public_targets if delay==nominal_delay
                              else np.array([pwm(m,t,delay) for t in lags]))
            actual = evaluate(plant_outcomes(seed,factor,actual_targets),desired)
            for idx,lag in enumerate(SEND_LAGS):
                a,b = one_outcome(actual,idx),one_outcome(actual,idx+1)
                key = f"{split}-{seed}-{profile}-{transport}-{lag}"
                row = {
                    "key":key,"split":split,"seed":seed,"phase_deg":m.phase,"turn_sign":m.sign,
                    "prefix_offset_deg":m.scenario.events[1][1],
                    "target_offset_deg":m.scenario.events[2][1],"profile":profile,"transport":transport,
                    "from_request_ms":lag*10,"branch_delay_ms":delay[2]*10,
                    "now_arrival_ms":10*(lag+delay[2]),"wait_arrival_ms":10*(lag+WAIT+delay[2]),
                    "now_ok":int(a.feasible),"wait_ok":int(b.feasible),
                    "outcome":label(a.feasible,b.feasible),
                    "now_recovery_ms":None if a.recovery is None else a.recovery*10,
                    "wait_recovery_ms":None if b.recovery is None else b.recovery*10,
                    "now_peak_deg":a.peak,"wait_peak_deg":b.peak,
                    "now_full_peak_deg":a.peak_all,"wait_full_peak_deg":b.peak_all,
                }
                labels.append(row)
                for name,val in predicted.items():
                    pa,pb = one_outcome(val,idx),one_outcome(val,idx+1)
                    output.append({
                        **row,"model":name,
                        "predicted_now_recovery_ms":None if pa.recovery is None else pa.recovery*10,
                        "predicted_wait_recovery_ms":None if pb.recovery is None else pb.recovery*10,
                        "predicted_now_peak_deg":pa.peak,"predicted_wait_peak_deg":pb.peak,
                        "predicted_now_full_peak_deg":pa.peak_all,
                        "predicted_wait_full_peak_deg":pb.peak_all,
                    })
    return labels, output


def choose_action(r, guard_ms):
    deadline = RECOVERY_TICKS*10
    for option in ("wait","now"):
        t = r[f"predicted_{option}_recovery_ms"]
        peak = r[f"predicted_{option}_full_peak_deg"]
        if t is not None and t+guard_ms<=deadline and peak<=TRANSIENT_LIMIT:
            return option
    return "abstain"


def performance(rows, guard_ms):
    action = [choose_action(r,guard_ms) for r in rows]
    false_wait = sum(a=="wait" and not r["wait_ok"] for a,r in zip(action,rows))
    false_now = sum(a=="now" and not r["now_ok"] for a,r in zip(action,rows))
    urgent_correct = sum(a=="now" and r["outcome"]=="urgent" for a,r in zip(action,rows))
    both_wait = sum(a=="wait" and r["outcome"]=="both_ok" for a,r in zip(action,rows))
    both_now = sum(a=="now" and r["outcome"]=="both_ok" for a,r in zip(action,rows))
    mae, bias = [], []
    for r in rows:
        for action_type in ("now","wait"):
            a=r[f"{action_type}_recovery_ms"]; b=r[f"predicted_{action_type}_recovery_ms"]
            if a is not None and b is not None:
                bias.append(b-a)
                mae.append(abs(b-a))
    return {
        "pairs":len(rows),"urgent":sum(r["outcome"]=="urgent" for r in rows),
        "both_ok":sum(r["outcome"]=="both_ok" for r in rows),
        "wait_only":sum(r["outcome"]=="wait_only" for r in rows),
        "neither":sum(r["outcome"]=="neither" for r in rows),
        "unsafe_wait_cases":sum(not r["wait_ok"] for r in rows),
        "false_wait":false_wait,"false_now":false_now,"unsafe_actions":false_wait+false_now,
        "urgent_send":urgent_correct,"both_wait":both_wait,"premature_send":both_now,
        "abstain":sum(a=="abstain" for a in action),
        "mean_raw_recovery_error_ms":mean(mae) if mae else None,
        "mean_raw_bias_ms":mean(bias) if bias else None,
        "finite_recovery_comparisons":len(mae),
    }


def guard_selection(cal):
    chosen={}
    table=[]
    for model in MODEL_SPEC:
        rr = [x for x in cal if x["model"]==model]
        options=[]
        for g in (*range(0,201,10),800):
            p=performance(rr,g)
            # Shared risk cap covers false claims of feasible waiting on calibration.
            admissible=p["false_wait"] <= RISK_CAP*p["unsafe_wait_cases"]
            line={"model":model,"guard_ms":g,"admissible":int(admissible),**p}
            if admissible:
                options.append(line)
            table.append(line)
        chosen[model]=min(options,key=lambda a:a["guard_ms"])
    def rank(m):
        z=chosen[m]
        return (-(z["urgent_send"]+z["both_wait"]),z["unsafe_actions"],z["abstain"],m)
    small=min((m for m in MODEL_SPEC if not MODEL_SPEC[m][1]),key=rank)
    richer=min((m for m in MODEL_SPEC if MODEL_SPEC[m][1]),key=rank)
    return chosen,small,richer,table


def diagnose_factor_changes(heldout):
    entries=[]
    fixed=[r for r in heldout if r["transport"]=="fixed"]
    lookup={(r["profile"],r["seed"],r["from_request_ms"]):r for r in fixed}
    for p in PROFILES:
        if p=="nominal":
            continue
        for sign in (1,-1):
            shift=[]
            error=[]
            for r in fixed:
                if r["profile"]!=p or r["turn_sign"]!=sign:
                    continue
                z=lookup["nominal",r["seed"],r["from_request_ms"]]
                shift.append(r["outcome"]!=z["outcome"])
                if r["now_recovery_ms"] is not None and z["now_recovery_ms"] is not None:
                    error.append(r["now_recovery_ms"]-z["now_recovery_ms"])
            entries.append({"profile":p,"turn_sign":sign,"pairs":len(shift),
                            "label_changed_vs_nominal":sum(shift),
                            "mean_now_recovery_shift_ms":mean(error) if error else None})
    return entries


def decide_gate(perf, simple, memory):
    result={}
    for p in PROFILES:
        a=perf["fixed",p,simple]
        b=perf["fixed",p,memory]
        result[p]={
            "unsafe_action_change":b["unsafe_actions"]-a["unsafe_actions"],
            "unsafe_wait_change":b["false_wait"]-a["false_wait"],
            "urgent_send_change":b["urgent_send"]-a["urgent_send"],
            "allow_wait_change":b["both_wait"]-a["both_wait"],
        }
        r=result[p]
        r["gate"]=r["unsafe_action_change"]<=0 and r["unsafe_wait_change"]<=0 and \
                  r["urgent_send_change"]>0 and r["allow_wait_change"]>0
    return {"pass":result["nominal"]["gate"] and any(result[p]["gate"] for p in PROFILES[1:]),
            "decision_by_profile":result}


def manifest():
    return {
        "firmware":SOURCE_REVISION,
        "dt_ms":10,
        "offset_limit_deg":OFFSET_LIMIT,
        "band_deg":ERROR_BAND,
        "deadline_ms":RECOVERY_TICKS*10,
        "transient_cap_deg":TRANSIENT_LIMIT,
        "send_lags_ms":[x*10 for x in SEND_LAGS],
        "wait_ms":WAIT*10,
        "public_delay_ms":PUBLIC_DELAY*10,
        "transport_notes":"Fixed both arms at 80 ms. Jitter: one shared draw of 50–110 ms for the final "
                          "packet in both arms, separate draws for earlier packets. Sender predicts with 80 ms only.",
        "fit":list(FIT),
        "fit_assignment":"All 4 factors crossed with all 4 phase quadrants and both turn signs.",
        "calibration":list(CAL),
        "holdout":list(TEST),
        "shift_factors":{k:asdict(v) for k,v in HOLDOUT_FACTORS.items()},
        "fit_factors":[asdict(v) for v in FIT_FACTORS],
        "calibration_factors":[asdict(v) for v in CAL_FACTORS],
        "calibrated_false_wait_rate_cap":RISK_CAP,
        "gate_rule":"Holdout fixed transport: memory beats the stateless comparator chosen on calibration "
                    "on both correct urgent sends and waits when both feasible, with no increase in unsafe "
                    "actions or unsafe waits. Required on nominal and at least one shifted profile.",
    }


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"protocol.json",manifest())
    models=fit_models(out)
    print("Fit complete",flush=True)
    truth, preds=[],[]
    for split,seeds in (("calibration",CAL),("heldout",TEST)):
        for i,seed in enumerate(seeds):
            lab,pp=records_for_episode(i,seed,split,models)
            truth.extend(lab)
            preds.extend(pp)
        print(split,"replayed",flush=True)
    cal=[r for r in preds if r["split"]=="calibration"]
    held=[r for r in preds if r["split"]=="heldout"]
    selected, simple, memory, grid=guard_selection(cal)
    for r in preds:
        model=r["model"]
        r["calibrated_guard_ms"]=selected[model]["guard_ms"]
        r["raw_decision"]=choose_action(r,0)
        r["calibrated_decision"]=choose_action(r,r["calibrated_guard_ms"])
    write_csv(out/"paired_outcomes.csv",truth)
    write_csv(out/"paired_decisions.csv",preds)
    write_csv(out/"guard_calibration.csv",grid)
    true_test=[r for r in truth if r["split"]=="heldout"]
    factor=diagnose_factor_changes(true_test)
    write_csv(out/"one_factor_effects.csv",factor)
    result=[]
    lookup={}
    risk_curve=[]
    for transport,profile,model in product(("fixed","jitter"),PROFILES,MODEL_SPEC):
        rows=[r for r in held if r["transport"]==transport and r["profile"]==profile and r["model"]==model]
        floor=selected[model]["guard_ms"]
        for guard in range(0,161,10):
            risk_curve.append({"profile":profile,"transport":transport,"model":model,"guard_ms":guard,
                               **performance(rows,guard)})
        for version,guard in (("raw",0),("calibrated",floor)):
            a=performance(rows,guard)
            result.append({"profile":profile,"transport":transport,"model":model,"version":version,
                           "guard_ms":guard,**a})
            if version=="calibrated":
                lookup[transport,profile,model]=a
    compare=decide_gate(lookup,simple,memory)
    write_csv(out/"summary.csv",result)
    write_csv(out/"heldout_risk_sweep.csv",risk_curve)
    summary={"gate":compare,"selected_stateless":simple,"selected_with_memory":memory,
             "calibration_choices":selected,"models":{k:asdict(v) for k,v in models.items()},
             "one_factor_effects":factor}
    write_json(out/"report.json",summary)
    print(json.dumps(summary["gate"],ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"results")
    run(parser.parse_args().out)


