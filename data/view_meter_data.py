import pandas as pd
import sys

meters = pd.read_parquet("VEC_meter_number_data.parquet")
gens = pd.read_parquet("VEC_gen_meter_number_data.parquet")
print(meters.columns)

check_non_solar = [3078906, 39608, 3081194, 3093019]
isin = gens["Service Number"].isin(check_non_solar)
print(isin.sum())
print(len(gens))
