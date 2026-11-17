"""Render the three main plots using saved results."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
O = ROOT / "figures"
O.mkdir(exist_ok=True)
NAVY, TEAL, AMBER, GREY = "#355e82", "#057f7c", "#bc6244", "#8792a6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.spines.top":False,
                     "axes.spines.right":False,"text.color":"#1f344b","axes.labelcolor":"#1f344b",
                     "axes.edgecolor":"#a8b4c2","xtick.color":"#526579","ytick.color":"#526579",
                     "savefig.facecolor":"white","svg.fonttype":"none"})


def tab(file):
    with (R/file).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def data(file):
    return json.loads((R/file).read_text(encoding="utf-8"))


def save(fig, stem):
    for ext in ("png","svg"):
        fig.savefig(O/f"{stem}.{ext}",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def recovery():
    groups = ("nominal","loaded","asymmetric")
    one = data("g1_report.json")["validation"]
    two = data("g1_refit_report.json")["validation"]
    fig, ax = plt.subplots(1,2,figsize=(10.3,4.4),layout="constrained",sharey=True)
    for a, report, key, title in zip(ax,(one,two),("mean_recovery_error_s","recovery_mae_s"),
                                    ("Fit to angle trajectories","Fit to recovery time and errors")):
        q = np.arange(len(groups))
        for j,(model, name,c) in enumerate((("stateless", "Rate limit only",NAVY),
                                             ("memory", "Rate limit + velocity state",TEAL))):
            vals = [report[g][model][key]*1000 for g in groups]
            bars=a.bar(q+(j-.5)*.34,vals,width=.32,color=c,label=name)
            a.bar_label(bars,labels=[f"{v:.1f}" for v in vals],padding=3,fontsize=8.5)
        a.axhline(50,color=AMBER,ls="--",lw=1.15,label="50 ms send opportunity")
        a.set_xticks(q,["Nominal","Loaded","Asymmetric"])
        a.set_ylim(0,78)
        a.set_title(title,loc="left",fontweight="bold")
        a.grid(axis="y",color="#e5ebef",lw=.65)
        a.set_axisbelow(True)
    ax[0].set_ylabel("Mean absolute recovery-time error [ms]")
    ax[1].legend(loc="upper left",fontsize=8,frameon=False)
    fig.suptitle("Holdout recovery prediction · 36 alternatives per profile and fit",
                 fontweight="bold",fontsize=12.5)
    save(fig,"recovery_prediction")


def resources():
    rows=tab("communication_reference_summary.csv")
    fig, axs=plt.subplots(1,3,figsize=(11.3,4.35),sharey=True,layout="constrained")
    for a, profile in zip(axs,("nominal","loaded","asymmetric")):
        for name,col,mark,line in (("periodic",GREY,"o","Periodic"),
                                   ("threshold",NAVY,"s","Threshold"),
                                   ("threshold_dwell",TEAL,"D","Threshold + dwell")):
            matches=[r for r in rows if r["profile"]==profile and r["policy"]==name]
            matches.sort(key=lambda r:int(r["cap"]))
            x=[float(r["mean_updates"]) for r in matches]
            y=[int(r["passed"]) for r in matches]
            a.plot(x,y,color=col,marker=mark,ms=4.8,lw=1.6,label=line)
        p=next(r for r in rows if r["profile"]==profile and r["policy"]=="every_change")
        a.scatter(float(p["mean_updates"]),int(p["passed"]),color=AMBER,marker="^",s=49,
                  label="Every command change",zorder=5)
        a.set_title(profile.capitalize(),fontweight="bold")
        a.set_xlabel("Mean application updates per run")
        a.set_xticks([0,2,4,6,8,12,16])
        a.set_ylim(0,26)
        a.set_xlim(.7,16.8)
        a.grid(axis="y",color="#e5ebef",lw=.65)
    axs[0].set_ylabel("Tasks passed out of 24")
    axs[-1].legend(loc="lower right",frameon=False,fontsize=8)
    fig.suptitle("Communication references · identical board control and application messages",
                 fontweight="bold",fontsize=12.5)
    save(fig,"communication_resource_curves")


def witness():
    report=data("g0_report.json")
    if not report["witness"]:
        return
    diag=report["witness"][0]
    fig,ax=plt.subplots(3,1,figsize=(9,6.4),sharex=True,layout="constrained",
                       height_ratios=(1,1.15,1))
    ps=(("fresh","Every command change",GREY),("threshold","Tuned threshold",NAVY),
        ("retrospective","Retrospective, scripted",TEAL))
    for j,(k,name,c) in enumerate(ps):
        rows=tab("g0_witness_"+k+".csv")
        t=[float(x["t_s"]) for x in rows]
        if j==0:
            ax[0].step(t,[int(x["remote_deg"]) for x in rows],where="post",color="#1f344b",lw=1.45)
        ax[1].plot(t,[float(x["peak_abs_deg"]) for x in rows],color=c,lw=1.45,label=name)
        xs=[float(x["arrived_sent_s"]) for x in rows if x["arrived_sent_s"]]
        ax[2].eventplot(xs,lineoffsets=j,linelengths=.50,colors=c)
    ax[0].set_ylabel("Remote offset [deg]")
    ax[1].set_ylabel("Max angle error [deg]")
    ax[1].axhline(5,color=AMBER,ls=":",lw=1.1,label="5 deg recovery band")
    for a in ax:
        a.grid(axis="y",color="#e4e9ee",lw=.6)
    ax[2].set_yticks(range(3),[n for _,n,_ in ps])
    ax[2].set_xlabel("Time [s]")
    ax[2].set_xlim(.5,7.6)
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h,l,loc="outside lower center",frameon=False,fontsize=8,ncol=4)
    fig.suptitle(f"{diag['scene']} · fewer updates are possible for this synthetic command",
                 fontweight="bold",fontsize=12)
    save(fig,"screening_example")


if __name__=="__main__":
    recovery()
    resources()
    witness()


