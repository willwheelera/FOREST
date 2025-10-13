import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualize_network import visualize_network
import pickle
import loads_to_transformers
import read_in_data
import sys

FIGSIZE = (3, 3)
year0 = 2025
nyears = 20
GROWTH = "MH"
if len(sys.argv) > 1:
    GROWTH = sys.argv[1]
    assert GROWTH in ["MED", "MH", "HIGH", "base150MED", "base150MH", "base150HIGH", "tran150MED", "tran150MH", "tran150HIGH"]
failure_curves = pd.read_parquet(f"output/alburgh_tf_failure_curves_{GROWTH}_{year0}_{nyears}years.parquet")
fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
mapfile = "data/Alburgh/transformer_map.pkl" # map meters to transformers
#Ldata = pd.read_parquet(fname)
#with open(mapfile, "rb") as f:
#    load_map = pickle.load(f)["load_map"]
#m2t_map = loads_to_transformers.meter_to_transformer_matrix(load_map, Ldata.columns)
m2t_map = pd.read_parquet("data/Alburgh/transformer_map_matrix.parquet")
tf_devices = pd.read_parquet(f"output/alburgh_tf_devices_{GROWTH}_{year0}_{nyears}years_avg.parquet")
tf_ratings = read_in_data.read_transformer_ratings("Alburgh")

final_prob = failure_curves.iloc[-1]
inds = np.argsort(final_prob)
nmeters = m2t_map.sum(axis=0)[final_prob.index]

select = inds[-25:]

info = failure_curves.iloc[-1, select]
#print(info)
isort = tf_devices.index.sort_values()[-10:]
#print(tf_devices.loc["tran_87721"])
#print(tf_ratings.loc[isort])
tf_devices["nmeters"] = nmeters#[info.index]
print(m2t_map.shape)
not_tran = ~tf_devices.index.str.startswith("tran_")
tmp = tf_devices[not_tran]
print("adoption level")
#print(tmp[["nmeters", "hasH", "hasE", "hasS"]].sum(axis=0))
print(tmp[["nmeters", "hasH", "hasE", "hasS"]].sum(axis=0) / m2t_map.shape[0])

tf_devices = tf_devices.loc[info.index]
tf_devices["p_fail"] = info
tf_devices["ratedKVA"] = tf_ratings.loc[info.index, "ratedKVA"]
tf_devices["id"] = tf_ratings.loc[info.index, "id"]
tf_devices = tf_devices[["id", "p_fail", "nmeters", "ratedKVA", "H", "E", "S", "hasH", "hasE", "hasS"]]
print(tf_devices.round(3))


for seed in []:#, 5, 6, 7, 8, 9]:
    tmpdf = pd.read_parquet(f"output/alburgh_tf_devices_{GROWTH}_{year0}_{nyears}years_{seed}.parquet")
    print(tmpdf.loc[info.index])

select_curves = failure_curves.values[:, select]

plt.figure(figsize=FIGSIZE)
plt.plot(final_prob.values[inds], label="p_fail")
#plt.plot(nmeters.values[inds]/10, label="nmeters/10")
plt.axvline(x=len(inds)-len(select), lw=0.4, c="k")
#plt.axhline(y=0.05, lw=0.4, c="k")
plt.xlabel("transformer index")
plt.ylabel("final failure probability")
plt.savefig(f"figures/alburgh_tf_failure_final_{GROWTH}_{year0}_{nyears}years.pdf", bbox_inches="tight")

plt.figure(figsize=FIGSIZE)
t = np.arange(len(failure_curves)) / 8760
plt.plot(t[::24], select_curves[::24])
plt.ylim([-0.04, 1.0])
plt.xlabel("year")
plt.ylabel("failure probability")
plt.title("Average of 100 runs")
plt.savefig(f"figures/alburgh_tf_failure_{GROWTH}_{year0}_{nyears}years.pdf", bbox_inches="tight")

visualize_network(xfmrs=failure_curves.columns[select])

plt.show()

