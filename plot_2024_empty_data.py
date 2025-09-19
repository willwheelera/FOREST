import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformer_aging
import read_in_data
import device_data
import weather_data.load_data

fname_aging = "output/alburgh_tf_aging_2024.parquet"
fname_load = "output/alburgh_tf_load_2024.parquet"
fname_devices = "output/alburgh_tf_devices_2024.parquet"

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

plt.figure(figsize=(3.5, 3.5))
select_tfs = np.argsort(df.values[-1])[-6:]
for i in select_tfs:
    plt.semilogy(np.arange(8760), df.iloc[:, i].values, label=df.columns[i])
plt.xlabel("time (h)")
plt.ylabel("Aging (h)")
plt.legend()

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
plt.axhline(y=5)
plt.axhline(y=20.5)
plt.xlabel("transformer index (sorted)")
plt.ylabel("Aging (years)")
plt.savefig(base_out+ "_age_distribution.pdf", bbox_inches="tight")

select_tfs = inds[np.arange(1380, df.shape[1], 1)]
_tfs = df.columns[select_tfs]
t = np.arange(len(df)) 

plt.figure(figsize=(3.5, 3.5))
for i in select_tfs:
    plt.semilogy(t, df.iloc[:, i].values, label=df.columns[i])
plt.xlabel("time (years)")
plt.ylabel("Aging (years)")
plt.legend()
plt.savefig(base_out+"_age_v_time.pdf", bbox_inches="tight")


plt.figure(figsize=(3.5, 3.5))
p_fail = transformer_aging.failure_prob(df.values, eta=112, beta=3.5)
for i in select_tfs:
    plt.plot(t, p_fail[:, i])
plt.xlabel("time (years)")
plt.ylabel("Failure probability")
plt.savefig(base_out + "_failure_v_time.pdf", bbox_inches="tight")

tf_names = df.columns[select_tfs]
tf_ratings = read_in_data.read_transformer_ratings("Alburgh")

high_age_info = tf_devices.loc[tf_names]
high_age_info["age"] = df.iloc[-1][tf_names]
high_age_info["p_fail"] = p_fail[-1][select_tfs]
high_age_info["ratedKVA"] = tf_ratings.loc[tf_names]["ratedKVA"].values
#meter
print(high_age_info)

##### Plot the transformer load at high transformers

dfload = pd.read_parquet(fname_load)
plt.figure(figsize=(3.5, 3.5))
for i in _tfs:
    plt.plot(np.arange(8760), dfload[i].values, label=i)
plt.axhline(y=1, ls=":", c="k")
plt.xlabel("time (h)")
plt.ylabel("Load (% rating)")
plt.legend()
#plt.savefig(base_out+"_load_v_time.pdf", bbox_inches="tight")


plt.show()
