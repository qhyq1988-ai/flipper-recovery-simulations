"""Render the two main deadline checks from their certified parameters."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from boundary import Interval, Problem, worst

ROOT=Path(__file__).resolve().parent
FIG=ROOT/"figures"
DATA=ROOT/"results"
FIG.mkdir(exist_ok=True)
COLORS=("#365d82","#087f77","#ad5153")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "text.color":"#253952","axes.labelcolor":"#253952",
                     "axes.edgecolor":"#a2b1bf","savefig.facecolor":"white",
                     "svg.fonttype":"none"})


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(fig,name):
    for ext in ("png","svg"):
        fig.savefig(FIG/f"{name}.{ext}",dpi=175,bbox_inches="tight")
    plt.close(fig)


def problem(data):
    d=dict(data)
    for key in ("e0","gain","forward","reverse","delay"):
        d[key]=Interval(**d[key])
    return Problem(**d)


def draw_boundary():
    r=load(DATA/"report.json")
    stages=r["adaptive"]["stages"]
    color=COLORS[:2]
    names=("Prior range","After certified probe")
    ptns=(stages[0],stages[1])
    fig,ax=plt.subplots(1,2,figsize=(10.2,4.0),layout="constrained")
    for stage,name,c in zip(ptns,names,color):
        p=problem(stage["problem"])
        ss=[k*.002 for k in range(125,336)]
        ax[0].plot([s*1000 for s in ss],[worst(p,s)*1000 for s in ss],
                   color=c,lw=1.9,label=name)
        t=stage["last_send_s"]
        ax[0].plot(t*1000,worst(p,t)*1000,"o",color=c)
        ax[0].plot((t+.05)*1000,worst(p,t+.05)*1000,"x",color=c)
        ax[0].annotate(f"{t*1000:.0f} ms",(t*1000,worst(p,t)*1000),xytext=(-8,-15),
                         textcoords="offset points",ha="center",color=c)
    ax[0].axhline(750,color=COLORS[2],ls="--",lw=1.1,label="750 ms deadline")
    ax[0].set(xlabel="Send time after request [ms]",ylabel="Worst-case recovery time [ms]",
              title="Latest certified send and first failing slot")
    ax[0].legend(loc="upper left",frameon=False,fontsize=8)
    ax[0].grid(axis="y",color="#e5ecf1")
    tags=("Prior","Probe 1","Probe 2","Uncertified probe")
    for j,x in enumerate(stages):
        bounds=x["problem"]["forward"]
        ax[1].plot([bounds["lo"],bounds["hi"]],[j,j],"o-",color=color[min(1,j)],lw=3,ms=4)
    speed=r["adaptive"]["forward_speed_required_at_500_ms"]
    ax[1].axvline(speed,ls="--",color=COLORS[2],lw=1.1,label="Required at 500 ms")
    ax[1].set_yticks(range(len(tags)),tags)
    ax[1].invert_yaxis()
    ax[1].set(xlabel="Admissible forward speed cap [deg/s]",title="Set-membership identification")
    ax[1].legend(loc="center right",frameon=False,fontsize=8)
    fig.suptitle("Single-joint adaptive send boundary",fontweight="bold",fontsize=12.5)
    save(fig,"adaptive_boundary")


def draw_scope():
    file=DATA/"v4_scope_check.json"
    if not file.exists():
        return
    cases=load(file)["cases"]
    fig,ax=plt.subplots(figsize=(7.8,3.8),layout="constrained")
    xs=list(range(len(cases)))
    ax.bar([x-.19 for x in xs],[r["actual_max_error_at_deadline_deg"] for r in cases],
           width=.34,color=COLORS[0],label="Original control")
    ax.bar([x+.19 for x in xs],[r["best_possible_max_error_lb_deg"] for r in cases],
           width=.34,color=COLORS[1],label="Lower bound with relaxed commands")
    ax.axhline(5.,ls="--",lw=1.1,color=COLORS[2],label="5 deg band")
    ax.set_xticks(xs,["Nominal\nNow","Nominal\nWait","Combined shift\nNow","Combined shift\nWait"])
    ax.set(ylabel="Maximum joint error at deadline [deg]",ylim=(0,9.4))
    ax.set_title("Flipper v4 episode 45400 · 750 ms deadline",fontweight="bold")
    ax.legend(loc="upper left",frameon=False,fontsize=8.4)
    ax.grid(axis="y",color="#e5ecf1")
    ax.set_axisbelow(True)
    save(fig,"flipper_travel_limit")


if __name__=="__main__":
    draw_boundary()
    draw_scope()


