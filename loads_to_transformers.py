import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from read_in_data import read_transformer_ratings

def run():
    path = "data/Alburgh/"
    fname = path+"transformer_loads.parquet"
    ratings = read_transformer_ratings("Alburgh")
    with open(path+"transformer_map.pkl", "rb") as file:
        maps = pickle.load(file)
    a = maps["load_map"]

    # Look for xfmrs serving many meters
    if False:
        meters, xfmrs = list(zip(*a))
        xfmr, counts = np.unique(xfmrs, return_counts=True)
        df = pd.DataFrame(dict(xfmr=xfmr, counts=counts))
        df.sort_values(by="counts", inplace=True)
        print(df.tail(25))
        meter, mcounts = np.unique(meters, return_counts=True)
        df = pd.DataFrame(dict(meter=meter, counts=mcounts))
        df.sort_values(by="counts", inplace=True)
        print(df.head(5))
        print(df.tail(25))
        quit()

    if os.path.exists(fname):
        load_tfdf = pd.read_parquet(fname)
    else:
        loadfile = "2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
        df = pd.read_parquet(path+loadfile).astype(float)
        load_tfdf = meters_to_transformers(maps["load_map"], df)
        load_tfdf.to_parquet(path+"transformer_loads.parquet")

    if os.path.exists(fname.replace("load", "gen")):
        gen_tfdf = pd.read_parquet(fname.replace("load", "gen"))
    else:
        genfile = "2024-01-01_2024-12-31_South_Alburgh_Gen_corrected.parquet"
        df = pd.read_parquet(path+genfile).astype(float)
        gen_tfdf = meters_to_transformers(maps["gen_map"], df)
        gen_tfdf.to_parquet(path+"transformer_gens.parquet")

    tfdf = load_tfdf.subtract(gen_tfdf, fill_value=0.)
    ratings = ratings.loc[tfdf.columns]
    tfdf = (tfdf / ratings["ratedKVA"].values)

    #plot_margins(tfdf)
    week, indices = pick_highest_week(tfdf) 
    plot_highest_transformers(tfdf, ratings, indices, title=f"Week {week}")
    plt.savefig("figures/alburgh_highest_xfmrs_week.pdf", bbox_inches="tight")

    hour, indices = pick_highest_hour(tfdf) 
    plot_highest_transformers(tfdf, ratings, indices, title=f"Hour {hour}")
    plt.savefig("figures/alburgh_highest_xfmrs_hour.pdf", bbox_inches="tight")

    plt.show()

def plot_highest_transformers(tfdf, ratings, indices, title=""):
    plt.figure(figsize=(6, 3))
    tfdf = tfdf[indices]
    inds = tfdf.abs().max(axis=0).argsort()[-10:]
    data = tfdf.iloc[:, inds]
    plt.plot(data, label=ratings.index[inds])
    plt.axhline(y=1, ls=":")
    plt.axhline(y=-1, ls=":")
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.ylabel("transformer power / capacity")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)

def pick_highest_week(tfdf): 
    data = tfdf[:8736].values.reshape(52, 7, 24, -1)
    week = np.argmax(np.sum(data**2, axis=(1, 2, 3)))
    return week, slice(week*168, (week+1)*168)
    
def pick_highest_hour(tfdf): 
    data = tfdf[:8736].values.reshape(52, 7, 24, -1)
    hour = np.argmax(np.sum(data**2, axis=(0, 1, 3)))
    return hour, slice(hour, None, 24)
    
    

def plot_margins(tfdf):
    thresholds = (0.9, 1., 1.1, 2)
    thres = 1.
    values = np.abs(tfdf.values)
    xfmrs_over = (values > thres).sum(axis=1) # each hour, sum over xfmrs
    hours_over = [(values > t).sum(axis=0) for t in thresholds] # each xfmr, sum over hours
    print("nonzero", (values.sum(axis=0) > 0.).sum(), "/", len(tfdf.columns))
    print(f"threshold {thres}")
    print("all entries > cap", (values > thres).sum(), "/", values.size)
    print("xfmrs > cap", (xfmrs_over > 0).sum(), "/", values.shape[1])
    for ho, t in zip(hours_over, thresholds):
        print(f"# hours > {t}", (ho > 0).sum(), "/", values.shape[0])

    plt.figure(figsize=(4, 4))
    comb = reduce(sum, map(np.sqrt, hours_over))
    #comb = np.dot(thresholds, hours_over)
    _inds = np.lexsort((*hours_over[::-1], comb))
    sort_inds = _inds[hours_over[0][_inds] > 0]
    for ho, t in zip(hours_over, thresholds):
        tmp = ho[sort_inds]
        plt.stairs(tmp, np.arange(len(tmp)+1), label=t)
    plt.yscale("log")
    plt.xlabel("transformers")
    plt.ylabel(f"# hours over ")
    plt.legend()
    plt.savefig("figures/alburgh_transformers_hours_over.pdf", bbox_inches="tight")

    #fig, axs = plt.subplots(1, 3)
    #tmp = xfmrs_over[:8736].reshape(52, 7, 24)
    #axs[0].bar(np.arange(52), tmp.max(axis=(1, 2)))
    #axs[0].set_xlabel("week of year")
    #axs[1].bar(np.arange(7), tmp.max(axis=(0, 2)))
    #axs[1].set_xlabel("day of week")
    #axs[2].bar(np.arange(24), tmp.max(axis=(0, 1)))
    #axs[2].set_xlabel("hour of day")
    #axs[0].set_ylabel(f"# xfmrs over capacity {thres}")
    #plt.tight_layout()

    plt.show()

def meters_to_transformers(meter_map, loaddf):
    tfs = np.unique([m[1] for m in meter_map])
    tfdf = pd.DataFrame(data=0., index=loaddf.index, columns=tfs)
    for m in meter_map:
        if m[0] in loaddf.columns:
            tfdf.loc[:, m[1]] += loaddf[m[0]].values
    return tfdf


if __name__ == "__main__":
    run()
