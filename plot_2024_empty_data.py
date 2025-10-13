import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformer_aging
import read_in_data
import device_data
import weather_data.load_data
from visualize_network import visualize_network
import sys

def run(YEAR=2024):
    fname_aging = f"output/alburgh_tf_aging_{YEAR}.parquet"
    fname_load = f"output/alburgh_tf_load_{YEAR}.parquet"
    fname_devices = f"output/alburgh_tf_devices_{YEAR}.parquet"
    m2t_map = pd.read_parquet("data/Alburgh/transformer_map_matrix.parquet")

    run_id = 0
    #fname_aging = f"output/alburgh_tf_aging_2025_5years_{run_id}.parquet"
    #fname_devices = f"output/alburgh_tf_devices_{run_id}.parquet"
    base_out = fname_aging.replace("years", "y")
    base_out = base_out.replace("_aging", "")
    base_out = base_out.replace(".parquet", "")

    df = pd.read_parquet(fname_aging)
    tf_devices = pd.read_parquet(fname_devices)
    df.drop(columns="tran_87721", inplace=True)
    tf_devices.drop(index="tran_87721", inplace=True)
    print(df.shape)

    fig, axs = plt.subplots(2,1, constrained_layout=True, figsize=(7.5, 4.5))
    #select_tfs = np.argsort(df.values[-1])[-10:]
    inds = np.argsort(df.values[-1])
    select_tfs = inds[-10:]
    _tfs = df.columns[select_tfs]
    p_fail = transformer_aging.failure_prob(df.values, eta=112, beta=3.5)

    tf_names = df.columns[select_tfs]
    tf_ratings = read_in_data.read_transformer_ratings("Alburgh")

    tf_devices["id"] = tf_ratings.loc[tf_names, "id"]
    high_age_info = tf_devices.loc[tf_names]
    high_age_info["age"] = df.iloc[-1][tf_names]
    high_age_info["p_fail"] = p_fail[-1][select_tfs]
    high_age_info["ratedKVA"] = tf_ratings.loc[tf_names]["ratedKVA"].values
    high_age_info["nmeters"] = m2t_map.sum(axis=0)[tf_names]
    high_age_info.drop(columns=["cchp", "home charger", "solar"], inplace=True)
    high_age_info = high_age_info[["id", "age", "p_fail",  "ratedKVA", "nmeters"]]
    #meter
    print(high_age_info)

    t = np.arange(len(df)) 
    for i in select_tfs:
        axs[0].semilogy(np.arange(len(df)), df.iloc[:, i].values, label=df.columns[i])
    #plt.set_xlabel("time (h)")
    axs[0].set_ylabel("Aging (years)")
    axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1.1))
    axs[0].set_title(str(YEAR))
    #plt.savefig(base_out+"_aging_v_time.pdf", bbox_inches="tight")

    ##### Plot the transformer load at high transformers

    dfload = pd.read_parquet(fname_load)
    #plt.subplots(constrained_layout=True, figsize=(5.5, 3.5))
    for i in _tfs:
        axs[1].plot(np.arange(len(dfload)), dfload[i].values, label=high_age_info.loc[i,"ratedKVA"])
    axs[1].axhline(y=1, ls=":", c="k")
    axs[1].set_xlabel("time (h)")
    axs[1].set_ylabel("Load (% rating)")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1.1))
    plt.savefig(base_out+"_load_v_time.pdf", bbox_inches="tight")

    ##### Once per year data only
    ncopies = 20
    df = pd.DataFrame(data=np.outer(np.arange(1, ncopies+1), df.values[-1]), columns=df.columns, index=pd.RangeIndex(start=0, stop=ncopies))
    #df = pd.DataFrame(data=np.tile(df.values, (ncopies, 1)), columns=df.columns, index=pd.RangeIndex(start=0, stop=8760*ncopies))
    #for y in range(ncopies-1, 0, -1):
    #    df.iloc[8760*y:] += df.loc[8760*y-1]

    nyears = int(len(df) )
    print("nyears", nyears)
    select_curves = df.values
    #weights = 0.1**np.arange(nyears)
    #sort_cost = weights @ select_curves
    sort_cost = select_curves[-1]
    inds = np.argsort(sort_cost)

    plt.figure(figsize=(3.5, 3.5))
    for i in np.arange(len(df)):
        l = df.iloc[i].values
        plt.semilogy(l[inds])
    plt.axhline(y=1)
    plt.axhline(y=20.5)
    plt.xlabel("transformer index (sorted)")
    plt.ylabel("Aging (years)")
    plt.savefig(base_out+ "_age_distribution.pdf", bbox_inches="tight")


    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(7.5, 7.5))
    bigkeys = dfload.columns.intersection(read_in_data.BIGKEYS)
    for i in range(3):
        bk = bigkeys[i*10:i*10+10]
        axs[i].plot(dfload[bk], label=bk)
        axs[i].legend(loc="upper left", bbox_to_anchor=(1, 1))
    axs[0].set_title(f"bigkeys loads {YEAR}")

    #select_tfs = inds[np.arange(1380, df.shape[1], 1)]
    select_tfs = inds[-10:]
    _tfs = df.columns[select_tfs]
    t = np.arange(len(df)) 

    #plt.figure(figsize=(3.5, 3.5))
    #for i in select_tfs:
    #    plt.semilogy(t, df.iloc[:, i].values, label=df.columns[i])
    #plt.xlabel("time (years)")
    #plt.ylabel("Aging (years)")
    #plt.legend()
    #plt.savefig(base_out+"_age_v_time.pdf", bbox_inches="tight")
    #
    #
    #plt.figure(figsize=(3.5, 3.5))
    #for i in select_tfs:
    #    plt.plot(t, p_fail[:, i])
    #plt.xlabel("time (years)")
    #plt.ylabel("Failure probability")
    #plt.savefig(base_out + "_failure_v_time.pdf", bbox_inches="tight")


    print(tf_names)
    visualize_network(xfmrs=tf_names)

    plt.show()

if __name__ == "__main__":

    if len(sys.argv) > 1:
        for YEAR in sys.argv[1:]:
            run(YEAR=int(YEAR))

