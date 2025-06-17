import pandas as pd
import numpy as np

df = pd.read_parquet("gs_transformer060825.parquet")
df = df[~df["gs_substation"].isna()]
df = df[df["gs_substation"].astype(int) == 28]
df.reset_index()

#print(np.sort(df.columns.values))
print(df["gs_equipment_location"])
names = df["gs_equipment_location"]
names[names.isna()] = "None"

phases = [f"gs_tran_kva_{phase}" for phase in "abc"]
keys = phases + ["gs_rated_kva"]
for key in keys:
    none = df[key].isna()
    s = ~none & df[key].str.isnumeric()
    df.loc[~s, key] = 0.
    df[key] = df[key].astype(float)

maxs = df[phases].max(axis=1)
below = maxs > df["gs_rated_kva"]
print(df[keys][below])

newdf = df[["gs_equipment_location", "gs_rated_kva", "x", "y"]]
newdf.to_csv("Alburgh/2025_transformer_ratings.csv")

