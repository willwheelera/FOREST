import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.size"] = 7
matplotlib.rcParams["font.family"] = "serif"

def run(capF="_capF"):
    GROWTH = "MH"
    fname = f"../output/alburgh_tf_failure_curves_{GROWTH}_2025_20years.parquet"
    df = pd.read_parquet(fname)

    cost_pts = []
    cq_pts = []
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2))
    for n in [ 5, 7, 10, 20]:
        #upgrade = pd.read_parquet(f"opt_{GROWTH}_capF_{n}.parquet")
        if n==0:
            tf_sched = []
        else:
            tf_sched = optimized_schedule(f"opt_{GROWTH}{capF}_{n}.parquet")
        cost, cost_q, Efy = cost_function(df, tf_sched)
        cost_pts.append((n, cost))
        cq_pts.append((n, cost+cost_q))
        axs[0].plot(Efy, label=f"{n}")
    cost_pts = np.asarray(cost_pts).T
    cq_pts = np.asarray(cq_pts).T
    axs[1].plot(*cost_pts, label="w/o q")
    axs[1].plot(*cq_pts, label="w/ q")
    axs[0].set_xlabel("Year")
    axs[0].set_ylabel("Expected failures")
    axs[0].legend(title=r"$N^{\rm max}$")
    axs[1].legend()
    axs[1].set_xlabel(r"$N^{\rm max}$")
    axs[1].set_ylabel("Cost")
    plt.tight_layout()
    plt.savefig(f"../figures/schedule_optimization{capF}.pdf", bbox_inches="tight")
    
    
def cost_function(df, tf_sched):
    C_U = 5. # cost to upgrade (assume units of thousand$)
    C_D = 0.1 # value to defer each year
    C_F = 10. # cost of failure (including replacement with upgrade)
    C_EF = 1. # quadratic cost of simultaneous failure

    df = df.copy()
    for i, tfs in enumerate(tf_sched):
        df.loc[8760*(i+1)-1:, tfs] = 0.

    #upgrades = np.sum(tfdf["year"].values != 0)
    #deferral =  np.sum(tfdf["year"].values)
    #F1 = np.sum([df.loc[8760*(row["year"]+1)-1, row["tf"]] for row in tfdf.iterrows()])
    #noupgrade = tfdf["year"] == 0
    #F2 = df[tfdf[noupgrade]["tf"]].iloc[-1].sum()

    upgrades = 0
    deferral = 0
    for i, tfs in enumerate(tf_sched):
        upgrades += len(tfs)
        deferral += (i+1) * len(tfs)

    noplan_fail = df.iloc[-1].sum()
    Efail = df.max(axis=0).sum()
    P = np.diff(df[8759::8760].values, axis=0)
    P = np.maximum(P, 0.)
    Efail_year = P.sum(axis=1)
    print("sum Ef^2", np.sum(Efail_year**2))
    #print(Efail_year)
    wo_q = upgrades * C_U - deferral * C_D + Efail * (C_F - C_U) + noplan_fail * C_U
    return wo_q, np.sum(Efail_year**2) * C_EF, Efail_year
    

def optimized_schedule(fname, nyears=20):
    df = pd.read_parquet(fname)
    tf_sched = [[] for n in range(nyears)]
    for y in range(nyears):
        select = df["year"] == y+1
        tf_sched[y] = df[select]["tf"].values
    return tf_sched

if __name__ == "__main__":
    run("")
    run("_capF")
    plt.show()
