import pandas as pd
import sys
import numpy as np

meters = pd.read_parquet("VEC_meter_number_data.parquet")
gens = pd.read_parquet("VEC_gen_meter_number_data.parquet")
print(meters.columns)

s = meters["Substation"]
s = s[~s.isna()]
vals, counts = np.unique(s, return_counts=True)
for v, c in zip(vals, counts):
    print(v, c)
quit()

check_non_solar = [3078906, 39608, 3081194, 3093019]
isin = gens["Service Number"].isin(check_non_solar)
print(isin.sum())
print(len(gens))
