"""Construct terminal-error minimizers of the independent-input relaxation."""
import json
import math
from pathlib import Path

import numpy as np

from audit_reachability import (DEADLINE, DT, EXAMPLE_SEED, HOLDOUT_FACTORS, PUBLIC_DELAY,
                                SEND_LAGS, TASK_START, V4, fixture, pwm, reference, replay_sends)


def check():
    heldout = json.loads((V4 / "results/protocol.json").read_text())["holdout"]
    maneuver = fixture(EXAMPLE_SEED, heldout.index(EXAMPLE_SEED))
    command = np.stack([pwm(maneuver, s, (PUBLIC_DELAY,)*3) for s in SEND_LAGS[:2]])
    arrivals = TASK_START + np.array(SEND_LAGS[:2]) + PUBLIC_DELAY
    desired = reference(maneuver.scenario)[DEADLINE]
    archive = Path(__file__).parent / "results/summary.json"
    bound = json.loads(archive.read_text())["illustrated_witness"]["offline_lower_bound_deg"]
    for name in ("nominal", "combined"):
        plant, _, states, velocities = replay_sends(EXAMPLE_SEED, HOLDOUT_FACTORS[name], command, arrivals)
        for branch, a in enumerate(arrivals):
            h = DT/2
            alpha = np.exp(-h/plant.lag)
            ticks = np.repeat(np.arange(a, DEADLINE), 2)
            scales = 1-.08*(1+np.sin(ticks[:, None]*DT*1.7+plant.phase))
            powers = np.arange(len(ticks), 0, -1)[:, None]
            weights = h*(1-alpha**powers)
            offset = states[branch] + h*(alpha**powers).sum(axis=0)*velocities[branch]
            lower, upper = -scales*plant.down, scales*plant.up
            qlo, qhi = (offset + (weights*requests).sum(axis=0) for requests in (lower, upper))
            optimum = np.clip(desired, qlo, qhi)
            mix = (optimum-qlo)/(qhi-qlo)
            q, v = states[branch].copy(), velocities[branch].copy()
            for request in lower+mix*(upper-lower):
                v = alpha*v+(1-alpha)*request
                q += h*v
            np.testing.assert_allclose(q, optimum, atol=1e-10, rtol=0)
            error = np.max(np.abs(q-desired))
            if name=="combined" and branch==1:
                assert math.isclose(error, bound, abs_tol=1e-10)
            print(name, ("early","late")[branch], f"{error:.6f} deg")


if __name__=="__main__":
    check()

