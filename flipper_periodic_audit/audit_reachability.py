"""Audit the offline reachable-angle test on the frozen Flipper held-out pairs."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "flipper_recovery_boundary_v4"
sys.path.insert(0, str(V4))

from experiment import (  # noqa: E402
    DT, END, HOLDOUT_FACTORS, PUBLIC_DELAY, SEND_LAGS, TASK_START, WAIT,
    WARM_TICKS, evaluate, fixture, jitter_seed, pwm, reference, servo_bank,
)
from platform_model import ERROR_BAND, RECOVERY_TICKS  # noqa: E402

DEADLINE = TASK_START + RECOVERY_TICKS
TICK_MS = round(1000 * DT)
EXAMPLE_SEED = 45400


def replay_sends(seed, factors, targets, adoption):
    """Batch the original simulator's six distinct send times."""
    plant = servo_bank(seed, factors)
    q = targets[:, 0, :].copy()
    velocity = np.zeros_like(q)
    gear = q.copy()
    angles = np.empty_like(targets, dtype=float)
    q_at_adoption = np.empty_like(q)
    v_at_adoption = np.empty_like(velocity)
    for i, cmd in enumerate(targets.transpose(1, 0, 2)):
        angles[:, i, :] = q
        adopt = adoption == i - WARM_TICKS
        if adopt.any():
            q_at_adoption[adopt] = q[adopt]
            v_at_adoption[adopt] = velocity[adopt]
        q, velocity, gear = plant.step(q, velocity, gear, cmd, i - WARM_TICKS)
    return plant, angles, q_at_adoption, v_at_adoption


def lower_bound(plant, q, velocity, adoption, target):
    """Propagate the two extremal request sequences from the adoption state."""
    endpoints = np.stack([q, q])
    speeds = np.stack([velocity, velocity])
    caps = np.stack([-plant.down, plant.up])
    beta = 1 - np.exp(-DT / (2 * plant.lag))
    for tick in range(adoption, DEADLINE):
        scale = 1 - .08 * (1 + math.sin(tick * DT * 1.7 + plant.phase))
        for _ in range(2):
            speeds += beta * (caps * scale - speeds)
            endpoints += DT * speeds / 2
    low, high = endpoints
    return float(np.max(np.maximum(0, np.maximum(low - target, target - high))))


def load_pairs():
    with (V4 / "results/paired_outcomes.csv").open(encoding="utf8", newline="") as handle:
        return {
            row["key"]: row
            for row in csv.DictReader(handle)
            if row["split"] == "heldout"
        }


def save_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit():
    protocol = json.loads((V4 / "results/protocol.json").read_text(encoding="utf8"))
    seeds = protocol["holdout"]
    archived = load_pairs()
    sends = (*SEND_LAGS, SEND_LAGS[-1] + WAIT)
    rows, traces = [], []
    for seed_index, seed in enumerate(seeds):
        maneuver = fixture(seed, seed_index)
        desired = reference(maneuver.scenario)
        for transport in ("fixed", "jitter"):
            delay = (PUBLIC_DELAY,) * 3 if transport == "fixed" else jitter_seed(seed)
            adoption = TASK_START + np.asarray(sends) + delay[2]
            targets = np.stack([pwm(maneuver, send, delay) for send in sends])
            for profile, factors in HOLDOUT_FACTORS.items():
                plant, angles, q_at_adoption, v_at_adoption = replay_sends(
                    seed, factors, targets, adoption,
                )
                error, recovery, peak, _, passed = evaluate(angles, desired)
                for i, send in enumerate(SEND_LAGS):
                    late = i + 1
                    key = f"heldout-{seed}-{profile}-{transport}-{send}"
                    original = archived.pop(key)
                    expected_adoption = (adoption[late] - TASK_START) * TICK_MS
                    assert expected_adoption == int(original["wait_arrival_ms"]), key
                    assert (int(passed[i]), int(passed[late])) == (
                        int(original["now_ok"]), int(original["wait_ok"]),
                    ), key
                    np.testing.assert_allclose(
                        peak[[i, late]], [float(original["now_peak_deg"]), float(original["wait_peak_deg"])],
                        atol=1e-9, rtol=0, err_msg=key,
                    )
                    late_error = float(error[late, DEADLINE])
                    eligible = int(adoption[late]) < DEADLINE
                    bound = None
                    if eligible:
                        bound = lower_bound(
                            plant, q_at_adoption[late], v_at_adoption[late],
                            int(adoption[late]), desired[DEADLINE],
                        )
                        if bound > late_error + 1e-9:
                            raise ValueError(f"Reachable-angle lower bound exceeds actual error for {key}")
                    rows.append({
                        "key": key, "seed": seed, "profile": profile, "transport": transport,
                        "early_send_ms": send * TICK_MS,
                        "early_adoption_ms": (int(adoption[i]) - TASK_START) * TICK_MS,
                        "late_adoption_ms": int(expected_adoption),
                        "early_pass": bool(passed[i]), "late_pass": bool(passed[late]),
                        "early_deadline_error_deg": float(error[i, DEADLINE]),
                        "late_deadline_error_deg": late_error,
                        "eligible_pre_deadline_adoption": eligible,
                        "offline_lower_bound_deg": bound,
                        "offline_excluded": bool(bound is not None and bound > ERROR_BAND),
                    })
                if seed == EXAMPLE_SEED and transport == "fixed" and profile in ("nominal", "combined"):
                    for branch, i in (("early", 0), ("late", 1)):
                        for tick in range(TASK_START, END + 1):
                            traces.append({
                                "profile": profile, "choice": branch,
                                "time_ms": (tick - TASK_START) * TICK_MS,
                                "maximum_joint_error_deg": float(error[i, tick]),
                            })
    assert not archived, "Some frozen held-out rows were not replayed"
    return rows, traces


def summarize(rows):
    eligible = [row for row in rows if row["eligible_pre_deadline_adoption"]]
    misses = [row for row in eligible if row["late_deadline_error_deg"] > ERROR_BAND]
    urgent = [row for row in misses if row["early_pass"]]
    return {
        "pairs": len(rows),
        "late_adoption_before_deadline": len(eligible),
        "late_adoption_at_or_after_deadline": len(rows) - len(eligible),
        "eligible_late_deadline_misses": len(misses),
        "eligible_misses_excluded_offline": sum(row["offline_excluded"] for row in misses),
        "eligible_early_pass_late_deadline_miss": len(urgent),
        "eligible_early_pass_late_deadline_miss_excluded_offline": sum(row["offline_excluded"] for row in urgent),
    }


def plot(rows, traces, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), layout="constrained")
    trace_ax, audit_ax = axes
    colors = {"nominal": "#276A9D", "combined": "#AE5034"}
    for profile in colors:
        for choice in ("early", "late"):
            record = [row for row in traces if row["profile"] == profile and row["choice"] == choice]
            trace_ax.plot(
                [row["time_ms"] for row in record],
                [row["maximum_joint_error_deg"] for row in record],
                c=colors[profile], ls="-" if choice == "early" else "--",
                linewidth=1.65, label=f"{profile.capitalize()} / {choice}",
            )
    trace_ax.axhline(ERROR_BAND, color="#5E6571", lw=1, ls=":")
    trace_ax.axvline(750, color="#343A43", lw=1, ls=":")
    ymax = max(row["maximum_joint_error_deg"] for row in traces if 400 <= row["time_ms"] <= 950)
    trace_ax.set(xlim=(400, 950), ylim=(0, ymax + 2.6), xlabel="Time from request (ms)",
                 ylabel="Maximum joint error (deg)", title="(a) Matched Flipper maneuver")
    for when in (530, 580):
        trace_ax.axvline(when, color="#9AA2AE", lw=.8)
        trace_ax.text(when, .97, str(when), transform=trace_ax.get_xaxis_transform(),
                      ha="center", va="top", color="#505869", fontsize=8)
    trace_ax.text(750, .97, "750 ms", transform=trace_ax.get_xaxis_transform(),
                  ha="center", va="top", color="#343A43", fontsize=8)
    trace_ax.text(945, ERROR_BAND+.3, "5 deg", ha="right", color="#505869", fontsize=8)
    trace_ax.legend(loc="upper right", bbox_to_anchor=(1, .88), frameon=False, fontsize=7.5)
    for bound_proves, label, color, marker in (
        (False, "Unresolved", "#6E8399", "o"),
        (True, "Excluded", "#AE5034", "^"),
    ):
        group = [row for row in rows if row["eligible_pre_deadline_adoption"]
                 and row["early_pass"] and row["late_deadline_error_deg"] > ERROR_BAND
                 and row["offline_excluded"] == bound_proves]
        audit_ax.scatter(
            [row["late_deadline_error_deg"] for row in group],
            [row["offline_lower_bound_deg"] for row in group],
            color=color, alpha=.68, marker=marker, s=20, label=f"{label} ({len(group)})",
        )
    x_max = max(row["late_deadline_error_deg"] for row in rows
                if row["eligible_pre_deadline_adoption"] and row["early_pass"])
    audit_ax.plot([ERROR_BAND, x_max + 1], [ERROR_BAND, x_max + 1], lw=.8, color="#B4BBC4", ls="--")
    audit_ax.axhline(ERROR_BAND, color="#5E6571", lw=1, ls=":")
    audit_ax.set(xlim=(4.8, x_max + 1), ylim=(-.2, x_max + 1),
                 xlabel="Simulated late error at the deadline (deg)",
                 ylabel="Offline lower bound (deg)", title="(b) Early pass, late deadline miss")
    audit_ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(folder / "flipper_periodic_audit.png", dpi=300)
    fig.savefig(folder / "flipper_periodic_audit.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--figures", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    rows, traces = audit()
    summary = summarize(rows)
    assert summary["pairs"] == 1600
    summary["by_transport"] = {
        transport: summarize([row for row in rows if row["transport"] == transport])
        for transport in ("fixed", "jitter")
    }
    summary["by_profile"] = {
        profile: summarize([row for row in rows if row["profile"] == profile])
        for profile in HOLDOUT_FACTORS
    }
    witness = next(row for row in rows if row["key"] == "heldout-45400-combined-fixed-45")
    assert abs(witness["offline_lower_bound_deg"] - 6.457435649594252) < 1e-9
    summary["illustrated_witness"] = witness
    save_csv(args.out / "paired_reachability.csv", rows)
    save_csv(args.out / "matched_trajectories.csv", traces)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    plot(rows, traces, args.figures)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
