"""Audit inputs, train/test separation, resources, and clean reproduction."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

from platform_model import CHANNELS, map_turn_input, replay, turn_body


def read_json(out, name):
    return json.loads((out / name).read_text(encoding="utf-8"))


def read_csv(out, name):
    with (out/name).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--compare", type=Path)
    args=parser.parse_args()
    out=args.out
    assert map_turn_input(0,4)==4 and map_turn_input(180,-6)==-6
    for c in range(-10,11):
        assert map_turn_input(int(turn_body(c)[4:]),991)==c
    p=replay(0,100)
    assert p.pwm_tick.shape[1:]==(21,CHANNELS)
    assert list(p.pwm_tick[240,10,[0,-1]])==[286,295]
    manifest=read_json(out,"manifest.json")
    groups=[manifest["g0_seeds"],*manifest["identification_seeds"].values(),
            manifest["g2_tuning_seeds"],*manifest["g2_holdout_seeds"].values()]
    follow=read_json(out,"g1_refit_protocol.json")
    groups.extend([follow["fit_seeds"],*follow["heldout_seeds"].values()])
    assert sum(len(s) for s in groups)==len({t for s in groups for t in s})
    g0=read_json(out,"g0_report.json")
    assert g0["gate"] and g0["totals"]==dict(zero_update=0,every_change=16,ideal_arrival=16,periodic_1s=0)
    assert len(g0["witness"])==3
    assert all(k["offline_metrics"]["updates"]==2 and k["threshold"]["updates"]==6
               and k["offline_metrics"]["passed"] for k in g0["witness"])
    assert len(read_csv(out,"g1_validation.csv"))==216
    assert len(read_csv(out,"g1_refit_validation.csv"))==216
    primary=read_json(out,"g1_report.json")
    refit=read_json(out,"g1_refit_report.json")
    assert not primary["gate"] and not refit["gate"]
    assert all(d["memory"]["mean_recovery_error_s"] > d["stateless"]["mean_recovery_error_s"]
               for d in primary["validation"].values())
    assert refit["validation"]["loaded"]["memory"]["recovery_mae_s"] > \
           refit["validation"]["loaded"]["stateless"]["recovery_mae_s"]
    rows=read_csv(out,"communication_reference_trials.csv")
    assert len(rows)==1008 and len({r["case"] for r in rows})==72
    for row in rows:
        cap=int(row["cap"])
        if cap:
            assert int(row["updates"])<=cap
    p=read_csv(out,"communication_reference_packets.csv")
    cnt, bts={}, {}
    for item in p:
        key=(item["case"],item["policy"],item["cap"])
        cnt[key]=cnt.get(key,0)+1
        bts[key]=bts.get(key,0)+len(item["application_body"])
        assert float(item["arrived_s"])>=float(item["sent_s"])
        assert map_turn_input(int(item["application_body"][4:]),991)==int(item["offset_deg"])
    for row in rows:
        key=(row["case"],row["policy"],row["cap"])
        assert cnt.get(key,0)==int(row["updates"]) and bts.get(key,0)==int(row["body_bytes"])
    ref=read_csv(out,"communication_reference_summary.csv")
    eight=[x for x in ref if x["policy"]=="threshold_dwell" and x["cap"]=="8"]
    assert len(eight)==3 and sum(int(x["passed"]) for x in eight)==72
    status=read_json(out,"stage_status.json")
    assert status["history_aware_trigger"]=="stopped at approved identification gate"
    print("Verified firmware endpoints, PWM counts, disjoint splits, recovery decision, and all message accounting.")
    if args.compare:
        expected=sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
        for name in expected:
            a, b = (out/name).read_bytes(), (args.compare/name).read_bytes()
            assert hashlib.sha256(a).digest()==hashlib.sha256(b).digest(), name
        print(f"{len(expected)} result files match the independent rerun byte for byte.")


if __name__=="__main__":
    main()


