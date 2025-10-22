import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from timer import Timer
import matplotlib
matplotlib.rcParams["font.size"] = 7
matplotlib.rcParams["font.family"] = "serif"
#matplotlib.rcParams["font.serif"] = "Times"

FIGSIZE = (4, 7)

def expected_failures():
    GROWTH = "MH"
    fname = f"output/alburgh_tf_failure_curves_{GROWTH}_2025_20years.parquet"
    df = pd.read_parquet(fname)

    for n in [5, 8, 10]:
        s_opt = optimized_schedule(f"optimization/opt_{GROWTH}_capF_{n}.parquet", nyears)
    

def run(GROWTH="MH", n=5):
    timer = Timer(20)
    fname = f"output/alburgh_tf_failure_curves_{GROWTH}_2025_20years.parquet"
    df = pd.read_parquet(fname)
    timer.print("df loaded")
    print(df.shape)
    print("initial expected failures", df.iloc[-1].sum())
    #tfselect = df.iloc[-1].sort_values().index[-200:]

    nyears = 20
    s_end = end_schedule(df, n, nyears)
    timer.print("end schedule")
    s_greedy = greedy_schedule(df, n, nyears)
    timer.print("greedy schedule")
    s_back = backwards_schedule(df, n, nyears)
    timer.print("backtrack schedule")
    s_opt = optimized_schedule(f"optimization/opt_{GROWTH}_capF_{n}.parquet", nyears)
    timer.print("optimized schedule")

    tfids = df.iloc[-1].sort_values().index[-400:].tolist()
    #for s in s_end: tfids.extend(s)
    #for s in s_greedy: tfids.extend(s)
    #for s in s_back: tfids.extend(s)
    #for s in s_opt: tfids.extend(s)
    tfselect = np.unique(tfids)

    #fig, ax = plt.subplots(1, 1, figsize=(3,3))
    #plot(ax, df, s_end, tfselect, "final")
    #ax.set_ylabel("Failure probability")
    #ax.set_xlabel("year")
    #ax.set_title("")
    #plt.savefig(f"figures/alburgh_tf_withupgradesEND_n{n}_{GROWTH}_2025_20years.pdf", bbox_inches="tight")

    fig, axs = plt.subplots(4, 2, figsize=FIGSIZE)
    plot(axs[0], df, s_end, tfselect, "final")
    plot(axs[1], df, s_greedy, tfselect, "greedy")
    plot(axs[2], df, s_back, tfselect, "backtrack")
    plot(axs[3], df, s_opt, tfselect, "optimized")
    axs[0,0].set_ylabel("Failure probability")
    #axs[1].set_ylabel("Failure probability")
    #axs[2].set_ylabel("Failure probability")
    for ax in axs[:-1]:
        ax[0].set_xticks([])
    axs[-1,0].set_xlabel("year")
    axs[-1,1].set_xlabel("year")
    axs[0, 1].set_title("Expected no. failures")
    plt.tight_layout()
    plt.savefig(f"figures/alburgh_tf_withupgrades_n{n}_{GROWTH}_2025_20years.pdf", bbox_inches="tight")
    timer.print("plot made")

    plt.show()
    

def plot(ax, df_, tf_sched, tfselect, label):
    P = df_[8759::8760].diff()
    df = df_[tfselect].copy()
    top100 = df.iloc[-1].sort_values().index[-100:]
    for i, tfs in enumerate(tf_sched):
        df.loc[8760*(i+1)-1:, tfs] = 0.
        P.loc[i:, tfs] = 0.
    top20after = df.iloc[-1].sort_values().index[-20:]
    
    maxprob = df.max(axis=0)
    Ef = maxprob.sum()
    cost = cost_function(df, tf_sched)
    print(f"{label} - \t expected failures {maxprob.sum()} \t cost {cost}")

    df1 = df[top100]
    df2 = df[top20after]
    t = np.arange(len(df)) / 8760
    ax[0].plot(t[::24], df1.values[::24], lw=0.7)
    ax[0].plot(t[::24], df2.values[::24], lw=0.7)
    ax[0].set_ylim([-0.04, 1.0])
    ax[0].set_title(f"{label}\n E(fail) {Ef:.2f}", y=1., pad=-14, fontsize="medium")
    ax[0].grid(axis="y", lw=0.5, c="lightgray")

    P = np.maximum(P.values, 0.)
    Efy = P.sum(axis=1)
    Efy = np.concatenate([[0], Efy])
    ax[1].plot(Efy)
    
def cost_function(df, tf_sched):
    C_U = 5. # cost to upgrade (assume units of thousand$)
    C_D = 0.1 # value to defer each year
    C_F = 10. # cost of failure (including replacement with upgrade)
    C_EF = 1. # quadratic cost of simultaneous failure
    upgrades = 0
    deferral = 0
    for i, tfs in enumerate(tf_sched):
        upgrades += len(tfs)
        deferral += (i+1) * len(tfs)

    Efail = df.max(axis=0).sum()
    P = np.diff(df[8759::8760].values, axis=0)
    P = np.maximum(P, 0.)
    Efail_year = P.sum(axis=1)
    print("sum Ef^2", np.sum(Efail_year**2))
    #print(Efail_year)
    return upgrades * C_U - deferral * C_D + Efail * C_F + np.sum(Efail_year**2) * C_EF
    

def optimized_schedule(fname, nyears=20):
    df = pd.read_parquet(fname)
    tf_sched = [[] for n in range(nyears)]
    for y in range(nyears):
        select = df["year"] == y+1
        tf_sched[y] = df[select]["tf"].values
    return tf_sched
    

def end_schedule(df, n, nyears=20): 
    # failure prob df (hours, tfs)
    # replacements per year
    tfs = df.iloc[-1].sort_values().index.values[::-1] # sorted highest to lowest
    ntotal = n*nyears
    tf_sched = tfs[:ntotal].reshape(nyears, n)
    return tf_sched

def greedy_schedule(df, n, nyears=20):
    fname = f"tmp_greedy_schedule_{n}.pkl"
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
    tf_sched = []
    #tfs = df.iloc[-1].sort_values().index
    #df = df[tfs] # only need to consider these ones
    for y in range(nyears):
        t = 8760*y-1
        tmp = df.iloc[t]
        inds = np.argpartition(-tmp.values, n)[:n]
        tfs = tmp.index[inds]
        #tfs = df.iloc[t].sort_values().index[-n:]
        tf_sched.append(tfs)
        df = df.drop(columns=tfs)
    with open(fname, "wb") as f:
        pickle.dump(tf_sched, f)
    return tf_sched

def backwards_schedule(df, n, nyears=20):
    tf_sched = []
    ntotal = n*nyears
    tfs = df.iloc[-1].sort_values().index[-ntotal:]
    df = df[tfs].copy() # only need to consider these ones
    for y in range(nyears, 0, -1): # iterate backwards
        t = 8760 * y - 1 # end of year (y-1) zero-based; year y one-based
        tfs = df.iloc[t].sort_values().index[:n]
        tf_sched.append(tfs)
        df = df.drop(columns=tfs)
    #tf_sched.append(df.columns.values)
    print(len(tf_sched), [len(t) for t in tf_sched])
    return tf_sched[::-1]


if __name__ == "__main__":
    for n in [5,10]:#, 6, 7, 8]:
        run(n=n)
