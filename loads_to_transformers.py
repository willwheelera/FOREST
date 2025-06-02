import pickle
import pandas as pd
import numpy as np
import os

def run():
    path = "data/Alburgh/"
    fname = path+"transformer_loads.parquet"
    ratings = pd.read_csv(path+"transformer_ratings.csv", index_col=0)
    print(ratings.columns)
    ratings.set_index("tf_name", inplace=True)
    print(ratings)
    if os.path.exists(fname):
        tfdf = pd.read_parquet(fname)
    else:
        loadfile = "2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
        df = pd.read_parquet(path+loadfile)
        df = df.astype(float)

        with open(path+"transformer_map.pkl", "rb") as file:
            maps = pickle.load(file)
        
        
        tfdf = meters_to_transformers(df, maps["load_map"])
        tfdf.to_parquet(path+"transformer_loads.parquet")

    ratings = ratings.loc[tfdf.columns]
    tfdf = (tfdf.T / ratings.values).T

    print(tfdf)
    print((tfdf.values.sum(axis=0) > 0.).sum(), "/", len(tfdf.columns))
    print("all entries > cap", (tfdf.values > 1.).sum(), "/", tfdf.values.size)
    print("hours > cap", ((tfdf.values > 1.).sum(axis=1) > 0).sum(), "/", tfdf.values.shape[0])
    print("xfmrs > cap", ((tfdf.values > 1.).sum(axis=0) > 0).sum(), "/", tfdf.values.shape[1])

def meters_to_transformers(df, meter_map):
    tfs = np.unique([m[1] for m in meter_map])
    tfdf = pd.DataFrame(data=0., index=df.index, columns=tfs)
    for m in meter_map:
        if m[0] in df.columns:
            tfdf.loc[:, m[1]] += df[m[0]].values
    return tfdf


if __name__ == "__main__":
    run()
