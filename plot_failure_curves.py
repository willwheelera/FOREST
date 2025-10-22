import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from visualize_network import visualize_network
import pickle
import loads_to_transformers
import read_in_data
import sys
matplotlib.rcParams["font.size"] = 7
matplotlib.rcParams["font.family"] = "serif"

FIGSIZE = (3, 3)
year0 = 2025
nyears = 20

def run(basename):

    fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    mapfile = "data/Alburgh/transformer_map.pkl" # map meters to transformers
    #Ldata = pd.read_parquet(fname)
    #with open(mapfile, "rb") as f:
    #    load_map = pickle.load(f)["load_map"]
    #m2t_map = loads_to_transformers.meter_to_transformer_matrix(load_map, Ldata.columns)
    m2t_map = pd.read_parquet("data/Alburgh/transformer_map_matrix.parquet")
    tf_ratings = read_in_data.read_transformer_ratings("Alburgh")

    fig, axs = plt.subplots(1, 5, figsize=(9, 2))

    for i, GROWTH in enumerate(["LOW", "MED", "HIGH", "XMH"]):
        g_name = dict(LOW="MED", MED="MH", HIGH="HIGH", XMH="MH")[GROWTH]
        tag = f"{basename}{g_name}_{year0}_{nyears}years"
        xtag = "_1-1000" if GROWTH=="XMH" else ""
        tf_devices = pd.read_parquet(f"output/alburgh_tf_devices_{tag}_avg{xtag}.parquet")
        failure_curves = pd.read_parquet(f"output/alburgh_tf_failure_curves_{tag}{xtag}.parquet")

        ############################################

        final_prob = failure_curves.iloc[-1]
        inds = np.argsort(final_prob)
        nmeters = m2t_map.sum(axis=0)[final_prob.index]

        select = inds[-25:]

        info = failure_curves.iloc[-1, select]
        isort = tf_devices.index.sort_values()[-10:]
        tf_devices["nmeters"] = nmeters#[info.index]
        not_tran = ~tf_devices.index.str.startswith("tran_")
        tmp = tf_devices[not_tran]
        print(m2t_map.shape)
        print("adoption level")
        print(tmp[["nmeters", "hasH", "hasE", "hasS"]].sum(axis=0) / m2t_map.shape[0])

        ############################################

        tf_devices = tf_devices.loc[info.index]
        tf_devices["p_fail"] = info
        tf_devices["ratedKVA"] = tf_ratings.loc[info.index, "ratedKVA"]
        tf_devices["id"] = tf_ratings.loc[info.index, "id"]
        tf_devices = tf_devices[["id", "p_fail", "nmeters", "ratedKVA", "H", "E", "S", "hasH", "hasE", "hasS"]]
        #print(tf_devices.round(3))

        select_curves = failure_curves.values[:, select]

        #plt.figure(figsize=FIGSIZE)
        nhide = 800
        t = np.arange(nhide, len(final_prob))
        axs[0].plot(t, final_prob.values[inds][nhide:], label=GROWTH+xtag)
        #plt.savefig(f"figures/alburgh_tf_failure_final_{GROWTH}_{year0}_{nyears}years.pdf", bbox_inches="tight")

        #plt.figure(figsize=FIGSIZE)
        a = "bcde"[i]
        t = np.arange(len(failure_curves)) / 8760
        axs[i+1].plot(t[::24], select_curves[::24], c="k", lw=0.5)
        axs[i+1].set_ylim([-0.04, 1.0])
        axs[i+1].set_xlabel("Year")
        axs[i+1].set_yticklabels([])
        axs[i+1].tick_params(axis='y', which='major', length=0)
        axs[i+1].set_title(f"({a})", weight="bold", fontsize="medium", loc="left")
        axs[i+1].grid(axis="y", lw=0.5, c="lightgray")
        #plt.savefig(f"figures/alburgh_tf_failure_{GROWTH}_{year0}_{nyears}years.pdf", bbox_inches="tight")

        #visualize_network(xfmrs=failure_curves.columns[select])

    axs[0].grid(axis="y", lw=0.5, c="lightgray")
    axs[0].axvline(x=len(inds)-len(select), lw=0.5, c="k")
    axs[0].set_xlabel("Transformer index")
    axs[0].set_ylabel("Failure probability")
    axs[0].set_title("(a)", weight="bold", fontsize="medium", loc="left")
    axs[0].legend()
    plt.tight_layout()
    #plt.savefig(f"figures/alburgh_tf_failure_{basename}_{year0}_{nyears}years.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    for basename in [""]:#, "base150", "tran150"]:
        run(basename)
