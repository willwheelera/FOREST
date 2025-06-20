import pandas as pd
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("/home/wwheele1/MAPLE_BST_Demo")
from AMI_Player_Tools.parse_ami_data import format_ami_data


def save_corrected_data(year=2024):
    t0 = time.perf_counter()
    from_prefix = "/home/wwheele1/_FOREST/South_Alburgh/data_files"
    loads = pd.read_parquet(f"{from_prefix}/{year}-01-01_{year}-12-31_South_Alburgh_Load.parquet")
    gens = pd.read_parquet(f"{from_prefix}/{year}-01-01_{year}-12-31_South_Alburgh_Gen.parquet")
    loadmeter = read_meter_data("data/VEC_meter_number_data.parquet")
    genmeter = read_meter_data("data/VEC_gen_meter_number_data.parquet")
    loadmeter = loadmeter[loadmeter["Substation"] == 28]
    genmeter = genmeter[genmeter["Substation"] == 28]
    printtime("loaded files", t0)

    loads, gens = format_ami_data(loads, gens, loadmeter, genmeter)
    printtime("formatted", t0)
    
    loads.to_parquet(f"data/Alburgh/{year}-01-01_{year}-12-31_South_Alburgh_Load_corrected.parquet")
    gens.to_parquet(f"data/Alburgh/{year}-01-01_{year}-12-31_South_Alburgh_Gen_corrected.parquet")
    printtime("saved", t0)

def read_meter_data(fname):
    # Convert Substation to int and remove None values
    df = pd.read_parquet(fname)
    select = df["Substation"].str.isnumeric().fillna(False)
    df = df[select]
    df["Substation"] = df["Substation"].astype(int)
    return df
    
def filter_key(df, meterdf, remove):
    # what form of df? pivot with columns=asset_id
    rm_meters = meterdf[meterdf[remove]]["Meter Number"].astype(int)
    select = ~df.columns.astype(int).isin(rm_meters)
    cols = df.columns[select]
    return df[cols]
    
    
def printtime(s, t0): 
    print(s, round(time.perf_counter()-t0, 2), flush=True)

###########################################
# unsure if we need anything below

def remove_zeros(df, *keys, tol=1e-6):
    """df is AMI timeseries data"""
    # do each row instead of total per asset_id
    nonzero = np.ones(len(df), dtype=bool)
    for k in keys:
        nonzero = nonzero & (df[k].abs() > tol)
    return df[nonzero]
    

def compute_groupby(path, fname):
    loads = pd.read_parquet(path+fname)
    month_avg = loads.groupby(["asset_id", "month", "hour"]).mean()
    month_avg.reset_index(inplace=True)
    month_avg.to_parquet(path+fname.replace("load_difference", "month_avg_diff"))
    

def get_ami_year_difference():
    columns = ["asset_id", "start_date_time", "value"]
    t0 = time.perf_counter()

    path = "South_Alburgh/data_files/"
    loads2017 = load_parquet(path+"2017-01-01_2017-12-31_South_Alburgh_Load.parquet", columns=columns)
    loads2024 = load_parquet(path+"2024-01-01_2024-12-31_South_Alburgh_Load.parquet", columns=columns)

    ids17 = loads2017["asset_id"].unique()
    ids24 = loads2024["asset_id"].unique()
    common = set(ids24).intersection(ids17)
    common = list(common)#[:100] # just test
    loads2017 = df_select(loads2017, "asset_id", common)
    loads2024 = df_select(loads2024, "asset_id", common)

    print_time("selected common asset_ids", t0)

    loads = loads2017.merge(loads2024, on=["asset_id", "month", "day", "hour"], suffixes=("2017", "2024"))
    loads["difference"] = loads["value2024"] - loads["value2017"]
    print_time("joined dfs", t0)

    loads.to_parquet(path+"load_difference_2024-2017.parquet")

def df_select(df, key, select_set):
    return df[df[key].isin(select_set)]

def load_parquet(fname, columns=None):
    t0 = time.perf_counter()
    df = pd.read_parquet(fname, columns=columns)
    print_time(f"loaded {fname}", t0) 
    df["start_date_time"] = pd.to_datetime(df["start_date_time"])
    index = pd.DatetimeIndex(df["start_date_time"])
    print_time(f"made datetime index", t0) 
    df = df[index.minute == 0]
    print_time(f"selected 00 minutes", t0) 
    index = pd.DatetimeIndex(df["start_date_time"])
    df["month"] = index.month
    df["day"] = index.day
    df["hour"] = index.hour
    print_time(f"created year month day hour columns", t0) 
    df.drop(columns="start_date_time", inplace=True)
    return df

def print_time(s, t0):
    t = time.perf_counter() - t0
    print(s, np.around(t, 3), flush=True)



if __name__ == "__main__":
    save_corrected_data(2017)
    save_corrected_data(2019)
    save_corrected_data(2022)
