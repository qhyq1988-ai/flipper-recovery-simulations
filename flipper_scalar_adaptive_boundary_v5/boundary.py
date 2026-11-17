"""Exact single-joint deadline bounds and set-membership updates."""
from dataclasses import dataclass
from math import exp, log


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("Empty interval")

    def meet(self, other):
        return Interval(max(self.lo,other.lo),min(self.hi,other.hi))


@dataclass(frozen=True)
class Problem:
    step: float
    band: float
    deadline: float
    period: float
    e0: Interval
    gain: Interval
    forward: Interval
    reverse: Interval
    delay: Interval

    def __post_init__(self):
        if self.gain.lo<=0 or min(self.forward.lo,self.reverse.lo)<=0 or self.delay.lo<0:
            raise ValueError("Positive gain and speed bounds and nonnegative delay are required")
        if not (0<self.band<min(self.step,self.e0.lo)) or self.deadline<=0 or self.period<=0:
            raise ValueError("The initial and old-command gaps must exceed a positive band")
        if self.e0.hi<self.step:
            raise ValueError("Exact rectangular corner requires e0.upper >= step")


def remaining_gap(t, gap, k, speed):
    boundary=speed/k
    if gap<=boundary:
        return gap*exp(-k*t)
    linear=(gap-boundary)/speed
    if t<=linear:
        return gap-speed*t
    return boundary*exp(-k*(t-linear))


def time_to_band(e, band,k,speed):
    if e<=band:
        return 0.
    cutoff=speed/k
    if cutoff<=band:
        return (e-band)/speed
    if e<=cutoff:
        return log(e/band)/k
    return (e-cutoff)/speed + log(cutoff/band)/k


def completion(send, delay, e0,k,forward,reverse,step,band):
    a=send+delay
    if e0>=step:
        gap=step+remaining_gap(a,e0-step,k,forward)
    else:
        gap=step-remaining_gap(a,step-e0,k,reverse)
    return a+time_to_band(gap,band,k,forward)


def worst(p,send):
    return completion(send,p.delay.hi,p.e0.hi,p.gain.lo,p.forward.lo,p.reverse.lo,p.step,p.band)


def continuous_boundary(p):
    if worst(p,0)>p.deadline:
        return None
    lo,hi=0.,p.deadline
    for _ in range(72):
        if hi-lo<=1e-14:
            break
        mid=(lo+hi)/2
        if worst(p,mid)<=p.deadline:
            lo=mid
        else:
            hi=mid
    return lo


def latest_slot(p):
    slots=range(int(p.deadline/p.period+1e-11)+1)
    return max((i for i in slots if worst(p,i*p.period)<=p.deadline+1e-12),default=None)


def speed_update(prior,k_min,start_measured_gap,end_measured_gap,duration,error):
    """Update only after a whole sample interval has been certified saturated.

    The actual command is known and constant on this interval. If the measured
    end gap minus its error is above prior.hi/k_min, comparison with the
    scalar monotone flow proves saturation throughout. The true direction and
    the elapsed time are known and the parameter is fixed.
    """
    if min(k_min,duration)<=0 or error<0 or prior.lo<=0:
        raise ValueError("Invalid sampling or parameter bounds")
    if end_measured_gap-error<prior.hi/k_min:
        return prior,"saturation_not_certified"
    displacement=start_measured_gap-end_measured_gap
    lo=(displacement-2*error)/duration
    hi=(displacement+2*error)/duration
    return prior.meet(Interval(lo,hi)),"updated"


def gain_update(prior,speed_lo,start_measured_gap,end_measured_gap,duration,error):
    """Optional update from a certified unsaturated segment."""
    if min(speed_lo,prior.lo,duration)<=0 or error<0:
        raise ValueError("Invalid sampling or parameter bounds")
    if start_measured_gap+error>speed_lo/prior.hi or min(start_measured_gap,end_measured_gap)-error<=0:
        return prior,"linear_regime_not_certified"
    lo=log((start_measured_gap-error)/(end_measured_gap+error))/duration
    hi=log((start_measured_gap+error)/(end_measured_gap-error))/duration
    return prior.meet(Interval(lo,hi)),"updated"


