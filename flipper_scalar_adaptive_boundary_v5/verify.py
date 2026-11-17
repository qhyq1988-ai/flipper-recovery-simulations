"""Verify analytical endpoint witnesses, adaptation, and the v4 replay."""
from pathlib import Path
import argparse
import hashlib
import json

from boundary import Interval, gain_update, speed_update
from simulate_boundary import BASE, worst

p=argparse.ArgumentParser()
p.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"results")
p.add_argument("--compare",type=Path)
a=p.parse_args()
r=json.loads((a.out/"report.json").read_text(encoding="utf8"))
g=r["grid"]
assert g["case_paths"]==3840 and g["false_safe_count"]==0
assert g["classification_disagreement"]==0
assert g["largest_timing_difference_s"]<5e-6
xs=r["adaptive"]["stages"]
assert [x["last_slot"] for x in xs]==[8,10,10,10]
for x,y in zip(xs,xs[1:]):
    assert y["continuous_s"]>=x["continuous_s"]
    assert x["problem"]["forward"]["lo"]<=y["problem"]["forward"]["lo"]
    assert x["problem"]["forward"]["hi"]>=y["problem"]["forward"]["hi"]
assert [x["passed"] for x in r["stress"]["cases"]]==[1,0,1,0,0,1]
assert r["stress"]["drift_aware_margin"]["last_slot"]==9
assert 92.<r["adaptive"]["forward_speed_required_at_500_ms"]<94.
assert r["adaptive"]["examples"][2]["update"]=="saturation_not_certified"
assert speed_update(BASE.forward,18,6,4,.02,.075)[0]==BASE.forward
assert gain_update(BASE.gain,65,0.03,.015,.06,.025)[0]==BASE.gain
assert worst(BASE,.40)<BASE.deadline<worst(BASE,.45)
bridge=a.out/"v4_scope_check.json"
if bridge.exists():
    b=json.loads(bridge.read_text(encoding="utf8"))
    c=b["cases"]
    assert [x["actual_pass"] for x in c]==[True,True,True,False]
    assert [x["travel_limit_proves_deadline_impossible"] for x in c]==[False,False,False,True]
    assert 6.45<c[-1]["best_possible_max_error_lb_deg"]<6.47
    assert all(x["actual_max_error_at_deadline_deg"]>=x["best_possible_max_error_lb_deg"]-1e-8 for x in c)
print("PASS: sharp boundary, nested updates, worst-case witnesses, and optional v4 travel check.")
if a.compare is not None:
    fs=sorted(a.out.iterdir())
    for f in fs:
        assert hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256((a.compare/f.name).read_bytes()).digest(),f.name
    print(f"PASS: {len(fs)} result files match the clean rerun byte for byte.")

