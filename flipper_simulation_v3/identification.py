"""Identification from command/output logs. NumPy is the only solver dependency."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from statistics import mean, median
import csv
import numpy as np

from platform_model import DT, CHANNELS, WARM_TICKS, ERROR_BAND, RECOVERY_TICKS, VirtualServo


@dataclass(frozen=True)
class Predictor:
    gain: float
    speed: float
    velocity_memory_s: float

    @property
    def velocity_fraction(self):
        if not self.velocity_memory_s:
            return 1.
        return 1. - np.exp(-DT / self.velocity_memory_s)

    def step(self, position, velocity, target):
        v_limit = np.clip(self.gain * (target - position), -self.speed, self.speed)
        velocity = velocity + self.velocity_fraction * (v_limit - velocity)
        return position + DT * velocity, velocity

    def run(self, targets, initial=None):
        q = targets[0].copy() if initial is None else np.asarray(initial, dtype=float).copy()
        v = np.zeros_like(q)
        loaded = np.empty_like(targets, dtype=float)
        vel = np.empty_like(targets, dtype=float)
        for i, target in enumerate(targets):
            loaded[i] = q
            vel[i] = v
            q, v = self.step(q, v, target)
        return loaded, vel


def load_logs(paths):
    """CSV header t_s, pwm0...pwm11, q0...q11. Use uniform 10-ms resampling."""
    logs = []
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for i, row in enumerate(rows):
            if abs(float(row["t_s"])-i*DT) > 1e-7:
                raise ValueError(f"{path}: resample each complete trace to the 10-ms grid")
        if len(rows) < 101:
            raise ValueError(f"{path}: each usable trace must exceed one second")
        pwm = np.array([[float(x[f"pwm{i}"]) for i in range(CHANNELS)] for x in rows])
        angle = np.array([[float(x[f"q{i}"]) for i in range(CHANNELS)] for x in rows])
        if not np.isfinite(pwm).all() or not np.isfinite(angle).all():
            raise ValueError(f"{path}: nonfinite commands or angle data")
        logs.append((pwm, angle))
    return logs


def fit(logs, memory):
    length = min(len(q) for _, q in logs)
    t = np.array([u[:length] for u, _ in logs])
    y = np.array([v[:length] for _, v in logs])
    grid = list(product((9, 12, 16, 20, 23, 26, 30, 34, 38),
                        (95, 115, 135, 155, 175, 205),
                        (.009, .016, .023, .030, .039, .048, .060) if memory else (0.,)))
    gains, speeds, tau = (np.array(x)[:, None, None] for x in zip(*grid))
    fractions = 1 - np.exp(-DT / np.where(tau > 0, tau, 1.e-100))
    q = np.tile(y[:, 0, :], (len(grid), 1, 1))
    v = np.zeros_like(q)
    e = np.zeros(len(grid))
    for i in range(length):
        if i >= 100:
            e += np.mean((q - y[:, i, :]) ** 2, axis=(1, 2))
        wish = np.clip(gains * (t[:, i, :] - q), -speeds, speeds)
        v += fractions * (wish - v)
        q += DT * v
    idx = int(np.argmin(e))
    return Predictor(*grid[idx]), float(np.sqrt(e[idx] / (length - 100)))


def recovery_time(e, begin, end):
    data = e[begin:end]
    late = np.flatnonzero(data > ERROR_BAND)
    if not len(late):
        return 0
    if late[-1] + 1 == len(data):
        return None
    return int(late[-1] + 1)


def error_measures(angle, desired, end, begin):
    e = np.max(np.abs(angle - desired), axis=1)
    recovery = recovery_time(e, begin, end)
    worst = float(np.max(e[begin + RECOVERY_TICKS:end]))
    return e, recovery, worst, int(worst <= ERROR_BAND)



def main():
    import argparse
    import glob
    import json

    parser = argparse.ArgumentParser(description="Fit two angle-response models using complete 10-ms traces.")
    parser.add_argument("logs", nargs="+", help="CSV files or glob patterns")
    parser.add_argument("--validation", nargs="+", default=[], help="holdout CSV files or glob patterns")
    args = parser.parse_args()

    def paths(patterns):
        result = []
        for pattern in patterns:
            found = sorted(glob.glob(pattern))
            if not found:
                parser.error(f"No files match {pattern}")
            result.extend(Path(x).resolve() for x in found)
        return tuple(dict.fromkeys(result))

    trained = paths(args.logs)
    held_out = paths(args.validation)
    if set(trained) & set(held_out):
        parser.error("Training and holdout lists must be disjoint")
    data = load_logs(trained)
    holdout = load_logs(held_out) if held_out else []
    report = {}
    for name, has_memory in (("stateless", False), ("memory", True)):
        model, rmse = fit(data, has_memory)
        report[name] = {"parameters":asdict(model), "fitting_rmse_deg":rmse}
        if holdout:
            report[name]["holdout_rmse_deg_by_run"] = [
                float(np.sqrt(np.mean((model.run(u, y[0])[0][100:] - y[100:])**2)))
                for u,y in holdout
            ]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

