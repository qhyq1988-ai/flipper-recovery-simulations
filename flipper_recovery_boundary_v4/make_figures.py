"""Render the Flipper paired recovery results."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_results import read,load
from experiment import PROFILES

LABELS={"nominal":"Nominal","load":"Load","direction":"Direction","lag":"Drive lag","combined":"Combined"}
MODEL_LABELS={"sym_simple":"Symmetric limit, no memory",
              "sym_memory":"Symmetric limit + memory",
              "asym_simple":"Direction-specific limit, no memory",
              "asym_memory":"Direction-specific limit + memory"}
NAVY,TEAL,RED,GRAY="#365e83","#078279","#ab4c54","#8d9bac"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.spines.top":False,
                     "axes.spines.right":False,"text.color":"#213750","axes.labelcolor":"#213750",
                     "axes.edgecolor":"#a0aebd","xtick.color":"#456078","ytick.color":"#456078",
                     "savefig.facecolor":"white","svg.fonttype":"none"})


def save(fig,out,name):
    for ext in ("png","svg"):
        fig.savefig(out/f"{name}.{ext}",dpi=175,bbox_inches="tight")
    plt.close(fig)


def decisions(out,figs):
    table=read(out/"summary.csv")
    fig,axs=plt.subplots(1,3,figsize=(11.5,4.3),layout="constrained",sharex=True)
    columns=(("urgent_send","Correct urgent sends","urgent"),
             ("both_wait","Waits when both actions work","both_ok"),
             ("unsafe_actions","Actions that miss the task",None))
    for ax,(column,title,denom) in zip(axs,columns):
        q=np.arange(5)
        all_nums=[]
        models=tuple(MODEL_LABELS)
        colors=(GRAY,NAVY,RED,TEAL)
        for i,(m,c) in enumerate(zip(models,colors)):
            rows=[next(x for x in table if x["model"]==m and x["profile"]==p and
                              x["transport"]=="fixed" and x["version"]=="calibrated")
                  for p in PROFILES]
            a=[int(x[column]) for x in rows]
            all_nums.append(a)
            ax.bar(q+(i-1.5)*.18,a,.17,color=c,label=MODEL_LABELS[m])
            if i==0 and denom:
                den=[int(x[denom]) for x in rows]
        if denom:
            for x,n in zip(q,den):
                ax.text(x,max(lst[x] for lst in all_nums)+1.1,f"/{n}",
                        ha="center",fontsize=8,color="#456078")
        ax.set_xticks(q,[LABELS[p] for p in PROFILES],rotation=32,ha="right")
        ax.set(ylim=(0,max(max(a) for a in all_nums)+6),ylabel="Paired cases")
        ax.set_title(title,fontweight="bold",fontsize=10)
        ax.grid(axis="y",color="#e5ebef",lw=.7)
        ax.set_axisbelow(True)
    h,l=axs[0].get_legend_handles_labels()
    fig.legend(h,l,loc="outside lower center",frameon=False,ncol=2)
    fig.suptitle("Decisions at independently calibrated margins · 80 ms channel delay",
                 fontsize=12.5,fontweight="bold")
    save(fig,figs,"pair_decisions")


def context(out,figs):
    s=load(out/"diagnostic.json")
    v=s["example"]
    if v is None:
        return
    data=read(out/"illustrated_pair.csv")
    fig,axs=plt.subplots(1,2,figsize=(9.7,4.),layout="constrained",sharex=True,sharey=True)
    for ax,p in zip(axs,("nominal",v["profile"])):
        for action,c in (("now",TEAL),("wait",RED)):
            d=[x for x in data if x["profile"]==p and x["choice"]==action]
            t=[float(x["ms_from_request"]) for x in d]
            peak=[float(x["maximum_error_deg"]) for x in d]
            ax.plot(t,peak,color=c,lw=1.7,label="Send now" if action=="now" else "Wait 50 ms")
        ax.axvline(750,color=NAVY,ls=":",lw=1.1,label="Recovery deadline")
        ax.axhline(5,color="#5f7184",ls="--",lw=1.,label="Error band")
        ax.axvline(float(v["now_arrival_ms"]),color=TEAL,ls=":",lw=1)
        ax.axvline(float(v["wait_arrival_ms"]),color=RED,ls=":",lw=1)
        ax.set(xlim=(350,1060),xlabel="Time since new steering command [ms]",
               title=LABELS[p])
        ax.grid(axis="y",color="#e5ebef",lw=.65)
    axs[0].set_ylabel("Max joint error [deg]")
    h,l=axs[0].get_legend_handles_labels()
    fig.legend(h,l,loc="outside lower center",frameon=False,ncol=4,fontsize=8.3)
    fig.suptitle(f"Identical sender history and packet delay · episode {v['seed']}",
                 fontsize=12.5,fontweight="bold")
    save(fig,figs,"same_history_different_plant")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"results")
    p.add_argument("--figures",type=Path,default=Path(__file__).resolve().parent/"figures")
    a=p.parse_args()
    a.figures.mkdir(parents=True,exist_ok=True)
    decisions(a.out,a.figures)
    context(a.out,a.figures)

