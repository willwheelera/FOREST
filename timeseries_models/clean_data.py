import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import ami_data

def clean_all(df, meterdf=None):
    replace_hour(df) # daylight savings time skipped hour
    replace_hour(df, 71*24+6) # not sure the reason, anomalously low readings
    df = remove_solar(df, meterdf)
    df = remove_zeros(df)
    return df

def replace_hour(df, h=69*24+2): # default for daylight savings time hour
    df.values[h] = (df.values[h-1] + df.values[h+1]) / 2

def remove_solar(df, meterdf=None):
    if meterdf is None:
        meterdf = pd.read_parquet("quick_meter100.parquet")
    df = ami_data.filter_key(df, meterdf, "solar")
    return df

def remove_zeros(df):
    df = df.astype(float)
    iszero = df.sum(axis=0) == 0
    df.drop(columns=iszero[iszero].index.values, inplace=True)
    return df

def remove_all_devices(df, meterdf=None):
    if meterdf is None:
        meterdf = pd.read_parquet("quick_meter100.parquet")
    keys = ["cchp", "hpwh", "solar", "home charger"]
    nodevices = meterdf[keys].sum(axis=1) == 0
    meternums = meterdf[nodevices]["Meter Number"]
    select = df.columns[df.columns.astype(str).isin(meternums)]
    return df[select]
    
