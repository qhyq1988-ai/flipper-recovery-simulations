"""Reproduce the second scalar branch, boundary cases, and same-information baseline."""
import argparse
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

from boundary import Interval, completion, continuous_boundary, latest_slot, recovery_bounds, time_to_band
from simulate_boundary import BASE, numerical


def frozen_error_comparison():
    report = json.loads((Path(__file__).parent / "results/report.json").read_text())
    intervals = report["adaptive"]["stages"]
    calibrated = intervals[2]["problem"]["forward"]["lo"]
    rows = []
    for name, cap in (("prior", BASE.forward.lo), ("calibrated", calibrated),
                      ("drift", calibrated - 35)):
        p = replace(BASE, forward=Interval(cap, BASE.forward.hi))
        fixed = p.deadline - p.delay.hi - time_to_band(p.e0.hi, p.band, p.gain.lo, cap)
        fixed_slot = int(fixed / p.period) * p.period
        rows.append(dict(case=name, fixed_s=fixed, propagated_s=continuous_boundary(p),
                         fixed_slot_s=fixed_slot, propagated_slot_s=latest_slot(p)*p.period))
    assert [round(row["fixed_slot_s"]*1000) for row in rows] == [400, 500, 400]
    assert [round(row["propagated_slot_s"]*1000) for row in rows] == [400, 500, 450]
    return rows


def check_boxes():
    base = replace(BASE, deadline=.24, period=.025, delay=Interval(.01,.04),
                   gain=Interval(8,28), forward=Interval(100,155))
    rows = []
    boxes = [(name, replace(base, e0=e)) for name, e in
             (("below", Interval(7,10)), ("touching", Interval(14,18)), ("crossing", Interval(15,20)))]
    boxes.append(("reversed_gain", replace(base, e0=Interval(7,10), deadline=.272,
                                           gain=BASE.gain, forward=BASE.forward)))
    for name, p in boxes:
        initial = p.e0
        last = latest_slot(p)
        assert last is not None
        for slot in (last, last+1):
            send = p.period*slot
            low, high, witness = recovery_bounds(p, send)
            assert high-low <= 1.00001e-8
            if slot == last:
                assert high <= p.deadline
            else:
                assert low > p.deadline
                if name == "reversed_gain":
                    slow_gain_time = completion(send, p.delay.hi, initial.hi, p.gain.lo,
                                                p.forward.lo, p.reverse.hi, p.step, p.band)
                    assert witness == p.gain.hi and slow_gain_time < p.deadline
            gains = sorted({witness, *(p.gain.lo + i*(p.gain.hi-p.gain.lo)/20 for i in range(21))})
            largest_error = 0.
            count = 0
            for sign, e, k, vf, vo, delay in product(
                (-1,1), (initial.lo,initial.hi), gains,
                (p.forward.lo,p.forward.hi), (p.reverse.lo,p.reverse.hi), (p.delay.lo,p.delay.hi)
            ):
                exact = completion(send,delay,e,k,vf,vo,p.step,p.band)
                assert exact <= high+1e-12
                measured, _, _ = numerical(send,delay,e,k,vf,vo,sign,p,delta=.0005,horizon=.5)
                largest_error = max(largest_error, abs(measured-exact))
                count += 1
            assert largest_error < 5e-6
            rows.append(dict(case=name, slot_ms=round(1000*send), lower_s=low, upper_s=high,
                             witness_gain=witness, integrated=count, rk4_max_error_s=largest_error))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = json.dumps(dict(baselines=frozen_error_comparison(), branch_checks=check_boxes()), indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data+"\n",encoding="utf-8")
    print(data)

