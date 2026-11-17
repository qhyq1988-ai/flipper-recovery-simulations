"""Verify split integrity, paired exposures, and causal admissibility."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import hashlib

from analyze_results import read, load

p=argparse.ArgumentParser()
p.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"results")
p.add_argument("--compare",type=Path)
a=p.parse_args()
o=a.out
m=load(o/"protocol.json")
groups=[set(m[k]) for k in ("fit","calibration","holdout")]
assert [len(g) for g in groups]==[32,24,32]
assert len(set.union(*groups))==88
assert m["wait_ms"]==50 and m["offset_limit_deg"]==10 and m["deadline_ms"]==750
r=read(o/"paired_outcomes.csv")
assert len(r)==2800 and Counter(x["split"] for x in r)==dict(calibration=1200,heldout=1600)
lookup={x["key"]:x for x in r}
assert len(lookup)==len(r)
for x in r:
    assert int(x["wait_arrival_ms"])-int(x["now_arrival_ms"])==50
    assert int(x["now_arrival_ms"])-int(x["from_request_ms"])==int(x["branch_delay_ms"])
    assert int(x["branch_delay_ms"])==80 if x["transport"]=="fixed" else 50<=int(x["branch_delay_ms"])<=110
    assert abs(int(x["target_offset_deg"]))<=10 and abs(int(x["prefix_offset_deg"]))<=10
    label={("1","0"):"urgent",("1","1"):"both_ok",("0","1"):"wait_only",("0","0"):"neither"}
    assert x["outcome"]==label[x["now_ok"],x["wait_ok"]]
pred=read(o/"paired_decisions.csv")
assert len(pred)==4*len(lookup)
visible=defaultdict(set)
for x in pred:
    y=(x["key"],x["split"],x["seed"],x["profile"],x["transport"])
    assert y==(lookup[y[0]]["key"],lookup[y[0]]["split"],lookup[y[0]]["seed"],
               lookup[y[0]]["profile"],lookup[y[0]]["transport"])
    key=tuple(x[f] for f in ("split","seed","from_request_ms","model"))
    val=tuple(x[f] for f in ("predicted_now_recovery_ms","predicted_wait_recovery_ms",
                             "predicted_now_peak_deg","predicted_wait_peak_deg","calibrated_decision"))
    visible[key].add(val)
assert all(len(v)==1 for v in visible.values())
s=load(o/"report.json")
selection=s["calibration_choices"]
assert len(selection)==4
g=read(o/"guard_calibration.csv")
for name,x in selection.items():
    assert x["false_wait"] <= m["calibrated_false_wait_rate_cap"] * x["unsafe_wait_cases"]
    earliest=min(int(z["guard_ms"]) for z in g if z["admissible"]=="1" and z["model"]==name)
    assert earliest==x["guard_ms"]
assert not s["gate"]["pass"]
assert not s["gate"]["decision_by_profile"]["nominal"]["gate"]
assert s["gate"]["decision_by_profile"]["direction"]["gate"]
assert s["gate"]["decision_by_profile"]["combined"]["gate"]
d=load(o/"diagnostic.json")
assert d["example"]["same_input_nominal"]["outcome"]=="both_ok"
assert d["example"]["outcome"]=="urgent"
h=[z for z in pred if z["split"]=="heldout" and z["transport"]=="fixed"
   and z["from_request_ms"]==str(d["example"]["from_request_ms"]) and z["seed"]==d["example"]["seed"]]
assert len(h)==20
print("PASS: split isolation, matched packet arrivals, hidden-state information, and calibration rule.")
if a.compare:
    expected=[f for f in sorted(o.iterdir()) if f.is_file()]
    for f in expected:
        assert hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256((a.compare/f.name).read_bytes()).digest(),f.name
    print(f"PASS: {len(expected)} result files match a clean rerun byte for byte.")

