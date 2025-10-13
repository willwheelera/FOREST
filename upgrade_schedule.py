import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
from timer import Timer
matplotlib.rcParams["font.size"] = 7
matplotlib.rcParams["font.family"] = "serif"
#matplotlib.rcParams["font.serif"] = "Times"

FIGSIZE = (2, 5)

def run(GROWTH="MH", n=5):
    timer = Timer(20)
    fname = f"output/alburgh_tf_failure_curves_{GROWTH}_2025_20years.parquet"
    df = pd.read_parquet(fname)
    timer.print("df loaded")
    print(df.shape)
    print("initial expected failures", df.iloc[-1].sum())
    tfselect = df.iloc[-1].sort_values().index[-100:]

    nyears = 20
    s_end = end_schedule(df, n, nyears)
    timer.print("end schedule")
    s_greedy = greedy_schedule(df, n, nyears)
    timer.print("greedy schedule")
    s_back = backwards_schedule(df, n, nyears)
    timer.print("backtrack schedule")

    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    plot(ax, df, s_end, tfselect, "final")
    ax.set_ylabel("Failure probability")
    ax.set_xlabel("year")
    ax.set_title("")
    plt.savefig(f"figures/alburgh_tf_withupgradesEND_n{n}_{GROWTH}_2025_20years.pdf", bbox_inches="tight")

    fig, axs = plt.subplots(3, 1, figsize=FIGSIZE)
    plot(axs[0], df, s_end, tfselect, "final")
    plot(axs[1], df, s_greedy, tfselect, "greedy")
    plot(axs[2], df, s_back, tfselect, "backtrack")
    #axs[0].set_ylabel("Failure probability")
    axs[1].set_ylabel("Failure probability")
    #axs[2].set_ylabel("Failure probability")
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[-1].set_xlabel("year")
    plt.tight_layout()
    #plt.savefig(f"figures/alburgh_tf_withupgrades_n{n}_{GROWTH}_2025_20years.pdf", bbox_inches="tight")
    timer.print("plot made")

    plt.show()
    

def plot(ax, df, tf_sched, tfselect, label):
    df = df[tfselect].copy()
    for i, tfs in enumerate(tf_sched):
        df.loc[8760*(i+1)-1:, tfs] = 0.

    maxprob = df.max(axis=0)
    print(f"{label} - expected failures {maxprob.sum()}")

    t = np.arange(len(df)) / 8760
    ax.plot(t[::24], df.values[::24], lw=0.7)
    ax.set_ylim([-0.04, 1.0])
    ax.set_title(f"{label}", y=1., pad=-12, fontsize="medium")
    ax.grid(axis="y", lw=0.5, c="lightgray")
    

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
    df = df[tfs] # only need to consider these ones
    for y in range(nyears, 0, -1): # iterate backwards
        t = 8760 * y - 1 # end of year (y-1) zero-based; year y one-based
        tfs = df.iloc[t].sort_values().index[:n]
        tf_sched.append(tfs)
        df = df.drop(columns=tfs)
    #tf_sched.append(df.columns.values)
    print(len(tf_sched), [len(t) for t in tf_sched])
    return tf_sched[::-1]


if __name__ == "__main__":
    for n in [5]:#, 6, 7, 8]:
        run(n=n)
