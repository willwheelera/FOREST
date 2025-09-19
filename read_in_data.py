import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import weather_data.load_data
import device_data
import os
import time

def read_transformer_ratings(feeder="Alburgh", sub_id=False):
    path = f"data/{feeder}/"
    tf_fname = "normalized_transformer_loading.parquet"
    
    # Read in transformer rated kVA
    ratings = pd.read_csv(path+"transformer_ratings.csv", index_col=0)
    newratings = pd.read_csv(path+"2025_transformer_ratings.csv", index_col=0)
    ratings = ratings.set_index("tf_name")
    newratings = newratings.set_index("gs_equipment_location")

    # Try to fix incorrect values
    bigkeys = ['E72805100040038', 'E72805103079841', 'E72805103080326',
   'E72805103080328', 'E72805103097973', 'E72805203078657',
   'E72805203078706', 'E72805203078859', 'E72805203078866',
   'E72805203079531', 'E72805203079711', 'E72805203080317']
    ratings.loc[bigkeys, "ratedKVA"] = newratings.loc[bigkeys, "gs_rated_kva"]
    ratings.loc["E72805203096474", "ratedKVA"] = 100

    if sub_id:
        newratings = newratings[~newratings.index.isna()] # remove nan
        newratings = newratings[~newratings.index.duplicated(keep='first')] # remove duplicate names
        ratings["id"] = ratings.index
        shared_names = ratings.index.intersection(newratings.index)
        ratings.loc[shared_names, "id"] = newratings.loc[shared_names, "gs_facility_id"]
        ratings = ratings.set_index("id")
    return ratings


def load_transformer_data(mapfile, columns):
    with open(mapfile, "rb") as f:
        load_map = pickle.load(f)["load_map"]
    m2t_map = meter_to_transformer_matrix(load_map, columns)
    tf_ratings = read_transformer_ratings("Alburgh")
    tf_ratings = tf_ratings.loc[m2t_map.columns]
    return m2t_map, tf_ratings["ratedKVA"].values


def meter_to_transformer_matrix(meter_map, columns):
    meter_map = [l for l in meter_map if l[0] in columns]
    tfs = np.unique([m[1] for m in meter_map])
    mapdf = pd.DataFrame(data=0., index=columns, columns=tfs)
    for m in meter_map:
        mapdf.loc[m[0], m[1]] = 1.
    return mapdf

def load_meter_data(fname):
    Ldata = pd.read_parquet(fname)
    meterdf, keys = device_data.get_meter_data("sub28")
    #meterdf = meterdf[meterdf["Substation"] == 28]
    meterdf.set_index("Meter Number", inplace=True)
    Ldata = Ldata[Ldata.columns[Ldata.columns.isin(meterdf.index)]]
    Ldata = Ldata[:8760].astype(float)
    meterdf = meterdf.loc[Ldata.columns]
    return Ldata, meterdf

if __name__ == "__main__":
    read_transformer_ratings()
