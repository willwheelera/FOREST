import pandas as pd

fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Gen.parquet"
df = pd.read_parquet(fname)
print(df.columns)
print(df["asset_id"])
print(type(df["asset_id"][0]))
print((df["start_date_time"][0]))
print(type(df["start_date_time"][0]))
df["asset_id"] = df["asset_id"].astype(int)
df.drop(columns="channel")

df = pd.pivot_table(df, index="start_date_time", columns="asset_id", values="value")

df.to_parquet("data/Alburgh/test_gen.parquet")
