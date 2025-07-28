import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("output/alburgh_tf_aging_2025_10years.parquet")
print(df.shape)

nyears = int(len(df) / 8760)
select_curves = df.values[8759::8760]
#weights = 0.1**np.arange(nyears)
#sort_cost = weights @ select_curves
sort_cost = select_curves[0]

inds = np.argsort(sort_cost)
for i in np.arange(8759, len(df), 8760):
    l = df.iloc[i].values
    plt.semilogy(l[inds])

plt.figure()
for i in np.arange(0, df.shape[1], 200):
    plt.plot(df.iloc[::10, i].values)
plt.show()
