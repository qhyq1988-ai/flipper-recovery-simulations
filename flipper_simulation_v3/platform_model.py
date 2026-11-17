"""Firmware-level command replay and synthetic position-servo dynamics for Flipper."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

DT = 0.01
UPDATE_SLOT = 5
BOARD_SWEEP = 2
CHANNELS = 12
WARM_TICKS = 240
OFFSET_LIMIT = 10
AMP = 40.0
PWM_HZ = 50
PWM_COUNTS = 4096
RECOVERY_TICKS = 75
ERROR_BAND = 5.0
TRANSIENT_LIMIT = 26.0
SOURCE_REVISION = "ea0a30bfca2e1c8cf9df1145a588155065935d0c"


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    seed: int
    phase: int
    events: tuple[tuple[int, int], ...]
    horizon: int = 760
    profile: str = "nominal"
    delay: int = 8
    jitter: int = 3


def map_turn_input(raw: int, previous: int) -> int:
    """Native /params off path for the ordinary unsigned joystick domain."""
    if not 0 <= raw <= 180:
        raise ValueError("Joystick position must be in [0, 180]")
    mapped = (raw * 22) // 180 - 11
    return mapped if -OFFSET_LIMIT <= mapped <= OFFSET_LIMIT else previous


def turn_body(offset: int) -> bytes:
    """Common benchmark serialization, using the original off field."""
    if not -OFFSET_LIMIT <= offset <= OFFSET_LIMIT:
        raise ValueError("Unsupported effective offset")
    raw = round((offset + 11 + .5) * 180 / 22)
    assert map_turn_input(raw, 99) == offset
    return f"off={raw}".encode("ascii")


@dataclass(frozen=True)
class Replay:
    ideal_zero: np.ndarray
    pwm_angle: np.ndarray
    pwm_tick: np.ndarray
    writable: np.ndarray

    def targets(self, k: int, offset: int):
        return self.pwm_angle[k + WARM_TICKS, offset + OFFSET_LIMIT]


@lru_cache(maxsize=48)
def replay(phase: int, horizon: int, direction: int = 1) -> Replay:
    """Apply stock integer angle-to-PWM arithmetic, including channel 11 mapping.

    The 20-ms sweep and the two groups of I2C writes are assumed timings.
    The firmware waits 10 ms after writing all channels.
    """
    k = np.arange(-WARM_TICKS, horizon + 1)
    cycle = (phase + np.floor_divide(k, BOARD_SWEEP)) % 360
    temporal = (cycle * 3.1415 / 180.0).astype(np.float32)
    spatial = (np.arange(CHANNELS) * (2 * 3.1415 / (CHANNELS - 1)))[None, :]
    ideal_zero = AMP * np.sin(direction * temporal[:, None] + spatial)
    offset = np.arange(-OFFSET_LIMIT, OFFSET_LIMIT + 1)[None, :, None]
    integer_angle = np.trunc(90 + ideal_zero[:, None, :] + offset).astype(np.int32)
    span = np.array([1200] * 11 + [1300])
    pulse_width = 800 + (integer_angle * span) // 180
    pwm_tick = np.trunc(pulse_width.astype(np.float32) / 1_000_000 * PWM_HZ * PWM_COUNTS).astype(np.int16)
    pwm_angle = (pwm_tick.astype(float) * 1_000_000 / (PWM_HZ * PWM_COUNTS) - 800) * 180 / span - 90
    writable = (k % BOARD_SWEEP)[:, None] == (np.arange(CHANNELS) >= 6)[None, :]
    return Replay(ideal_zero, pwm_angle, pwm_tick, writable)


def steering_trace(scenario: Scenario) -> np.ndarray:
    c = np.zeros(scenario.horizon + 1, dtype=int)
    for t, value in scenario.events:
        assert t % UPDATE_SLOT == 0 and abs(value) <= OFFSET_LIMIT
        c[t:] = value
    return c


def on_board_offsets(n: int, arrivals: dict[int, int]) -> np.ndarray:
    r = np.zeros(n + 1, dtype=int)
    for t, offset in sorted(arrivals.items()):
        if 0 <= t <= n:
            r[t:] = offset
    return r


def board_trajectory(data: Replay, r: np.ndarray):
    """Return the continuous held PWM target and count for each joint and tick."""
    t = np.arange(-WARM_TICKS, len(r))
    b = np.r_[np.zeros(WARM_TICKS, dtype=int), r] + OFFSET_LIMIT
    pick = np.arange(len(t))[:, None] - ((t % BOARD_SWEEP)[:, None] != (np.arange(CHANNELS) >= 6))
    pick = np.maximum(0, pick)
    joints = np.arange(CHANNELS)[None, :]
    return data.pwm_angle[pick, b[pick], joints], data.pwm_tick[pick, b[pick], joints]


def delay_trace(scenario: Scenario) -> np.ndarray:
    return scenario.delay + np.random.default_rng(scenario.seed ^ 290831).integers(
        -scenario.jitter, scenario.jitter + 1, size=scenario.horizon + 1)


@dataclass
class VirtualServo:
    gain: np.ndarray
    lag: np.ndarray
    up: np.ndarray
    down: np.ndarray
    play: np.ndarray
    dead: np.ndarray
    trim: np.ndarray
    load: np.ndarray
    friction: np.ndarray
    phase: float

    @classmethod
    def create(cls, seed: int, profile: str) -> VirtualServo:
        r = np.random.default_rng(seed)
        shape = {
            "nominal": (1.00, 1.00, 1.00, 0.0),
            "loaded": (0.83, 1.24, 1.35, 0.8),
            "asymmetric": (0.92, 1.06, 1.60, 1.1),
        }
        speed, lag, backlash, extra = shape[profile]
        return cls(
            gain=r.uniform(23, 35, CHANNELS),
            lag=r.uniform(.022, .041, CHANNELS) * lag,
            up=r.uniform(142, 192, CHANNELS) * speed,
            down=r.uniform(138, 178, CHANNELS) * speed * (.75 if profile == "asymmetric" else 1.),
            play=r.uniform(.12, .40, CHANNELS) * backlash,
            dead=r.uniform(.03, .17, CHANNELS),
            trim=r.uniform(-.22, .22, CHANNELS),
            load=r.uniform(.25, 1.10, CHANNELS) + extra,
            friction=r.uniform(2., 5., CHANNELS),
            phase=float(r.uniform(-math.pi, math.pi)),
        )

    def step(self, q, velocity, gear, cmd, k: int):
        target = cmd + self.trim
        gear = np.clip(gear, target - self.play, target + self.play)
        load_scale = 1 - .08 * (1 + math.sin(k * DT * 1.7 + self.phase))
        disturbance = self.load * math.sin(k * DT * 2.0 + self.phase)
        for _ in range(2):
            error = gear - q
            residual = np.sign(error) * np.maximum(0.0, np.abs(error) - self.dead)
            request = self.gain * residual - self.friction * np.tanh(velocity / 9.0) + disturbance
            speed = np.clip(request, -self.down * load_scale, self.up * load_scale)
            velocity = velocity + (1 - np.exp(-DT / (2 * self.lag))) * (speed - velocity)
            q = q + DT * velocity / 2
        return q, velocity, gear


def physical_trace(scenario: Scenario, held: np.ndarray, pwm_targets: np.ndarray | None = None):
    data = replay(scenario.phase, scenario.horizon)
    pwm, count = board_trajectory(data, held)
    if pwm_targets is not None:
        pwm = pwm_targets
        count = None
    shape = pwm.shape
    q, v, gear = pwm[0].copy(), np.zeros(CHANNELS), pwm[0].copy()
    servo = VirtualServo.create(scenario.seed, scenario.profile)
    angles, velocity, train = np.empty(shape), np.empty(shape), np.empty(shape)
    for i, target in enumerate(pwm):
        angles[i], velocity[i], train[i] = q, v, gear
        q, v, gear = servo.step(q, v, gear, target, i - WARM_TICKS)
    return {
        "angle": angles,
        "velocity": velocity,
        "internal_gear": train,
        "pwm_angle": pwm,
        "pwm_tick": count,
    }


def reference(scenario: Scenario):
    return replay(scenario.phase, scenario.horizon).ideal_zero[WARM_TICKS:] + steering_trace(scenario)[:, None]


def score(scenario: Scenario, angle: np.ndarray, band: float = ERROR_BAND, recovery: int = RECOVERY_TICKS):
    error = np.max(np.abs(angle - reference(scenario)), axis=1)
    full_peak = float(np.max(error))
    peaks, settled, frames = [], [], []
    for i, (t, c) in enumerate(scenario.events):
        end = scenario.events[i + 1][0] if i + 1 < len(scenario.events) else scenario.horizon + 1
        if t + recovery >= end:
            continue
        part = error[t:end]
        peak = float(np.max(error[t+recovery:end]))
        late = np.flatnonzero(part > band)
        settle = int(late[-1] + 1) if len(late) else 0
        if settle == len(part):
            settle = None
        peaks.append(peak)
        settled.append(settle)
        frames.append((t, c, end, peak, settle))
    if not peaks:
        raise ValueError("Scenario has no scored task window")
    peak = max(peaks)
    return {
        "passed": int(peak <= band and full_peak <= TRANSIENT_LIMIT),
        "peak_scored": peak,
        "peak_all": full_peak,
        "eventual_recovery_s": max(settled) * DT if all(x is not None for x in settled) else None,
        "integral_error": float(np.sum(error) * DT),
    }, error, frames


