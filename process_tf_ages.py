import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformer_aging
import read_in_data
import device_data

#df = pd.read_parquet("output/alburgh_tf_aging_2024.parquet")
#tf_devices = pd.read_parquet("output/alburgh_tf_devices_2024.parquet")
run_id = 1
df = pd.read_parquet(f"output/alburgh_tf_aging_2025_5years_{run_id}.parquet")
tf_devices = pd.read_parquet(f"output/alburgh_tf_devices_{run_id}.parquet")
df.drop(columns="tran_87721", inplace=True)
tf_devices.drop(index="tran_87721", inplace=True)
print(df.shape)

nyears = int(len(df) / 8760)
select_curves = df.values[8759::8760]
#weights = 0.1**np.arange(nyears)
#sort_cost = weights @ select_curves
sort_cost = select_curves[-1]
inds = np.argsort(sort_cost)

plt.figure(figsize=(3.5, 3.5))
for i in np.arange(8759, len(df), 8760):
    l = df.iloc[i].values
    plt.semilogy(l[inds])
plt.axhline(y=5)
plt.axhline(y=20.5)
plt.xlabel("transformer index (sorted)")
plt.ylabel("Aging (years)")
plt.savefig(f"output/alburgh_tf_5y_{run_id}_age_distribution.pdf", bbox_inches="tight")

select_tfs = inds[np.arange(1380, df.shape[1], 1)]
t = np.arange(len(df)) / 8760

plt.figure(figsize=(3.5, 3.5))
for i in select_tfs:
    plt.semilogy(t[::10], df.iloc[::10, i].values)
plt.xlabel("time (years)")
plt.ylabel("Aging (years)")
plt.savefig(f"output/alburgh_tf_5y_{run_id}_age_v_time.pdf", bbox_inches="tight")

plt.figure(figsize=(3.5, 3.5))
p_fail = transformer_aging.failure_prob(df, eta=112, beta=3.5)
for i in select_tfs:
    plt.plot(t[::10], p_fail.iloc[::10, i].values)
plt.xlabel("time (years)")
plt.ylabel("Failure probability")
plt.savefig(f"output/alburgh_tf_5y_{run_id}_failure_v_time.pdf", bbox_inches="tight")

tf_names = df.columns[select_tfs]
tf_ratings = read_in_data.read_transformer_ratings("Alburgh")

high_age_info = tf_devices.loc[tf_names]
high_age_info["age"] = df.iloc[-1][tf_names]
high_age_info["p_fail"] = p_fail.iloc[-1][tf_names]
high_age_info["ratedKVA"] = tf_ratings.loc[tf_names]["ratedKVA"].values
#meter
#high_age_info["hasH"] = 
print(high_age_info)

plt.figure(figsize=(3.5, 3.5))
ax = plt.gca()
tf_devices[["H", "E", "S"]].iloc[inds].plot(ax=ax)
plt.xticks([])
plt.xlabel("transformer index (sorted)")
plt.ylabel("Total connected size")
plt.savefig(f"output/alburgh_tf_5y_{run_id}_devices.pdf", bbox_inches="tight")

plt.show()
