import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('future.no_silent_downcasting', True)



def get_meter_data():
    # who has heat pumps
    datapath = "/home/wwheele1/FOREST/data/"
    device_df = pd.read_excel(datapath+"All T3 through 2024.xlsx")
    clean_data(device_df)
    keys = ["cchp", "aev", "home charger", "hpwh", "centrally ducted", "mower", "phev", "battery"]
    meter_data = pd.read_parquet(datapath+"VEC_meter_number_data.parquet")
    clean_meter_numbers(meter_data)
    gen_data = pd.read_parquet(datapath+"VEC_gen_meter_number_data.parquet")
    add_device_data(device_df, meter_data, keys) # meter_data now contains boolean columns for each key
    add_solar(meter_data, gen_data)
    return meter_data, keys


def clean_meter_numbers(meterdf):
    for key in ["Meter Number", "Service Number"]:
        isdigit = meterdf[key].str.isnumeric().fillna(False).astype(bool)
        meterdf.drop(index=np.where(~isdigit)[0], inplace=True)
        meterdf.reset_index(inplace=True)

def add_solar(meter_data, gen_data):
    meter_data["solar"] = meter_data["Service Number"].isin(gen_data["Service Number"])

def clean_data(df):
    df["Measure"] = df["Measure"].str.lower()
    convert = {
        "mower - residential": "mower", 
        "induction stovetop": "induction cooktop", 
        "aev - income qualified": "aev",
        "phev - income qualified": "phev",
        "home charger - free": "home charger",
        "centrally ducted -commercial": "centrally ducted - commercial",
        "centrally ducted - residential": "centrally ducted",
    } 
    for k, v in convert.items():
        select = df["Measure"] == k
        df.loc[select, "Measure"] = v


def add_device_year_data(df, meterdf, keys): # full dataset
    df = df[[type(x) == int for x in df["Account"]]].copy()
    df["Service Number"] = df["Account"].values // 100

    meterdf.set_index("Service Number", inplace=True)
    measures = df[["Service Number", "Measure"]]
    # clean separately??
    #isdigit = meterdf["Service Number"].str.isdigit().fillna(False)
    #meterdf.drop(index=np.where(~isdigit)[0], inplace=True)
    for k in keys:
        meterdf[k] = 10000 # year infinity, later than any install
    for i, row in df.iterrows():
        k = row["Measure"]
        sn = row["Service Number"]
        meterdf.loc[sn, k] = row["Year"]
        #tmp = df[df["Measure"] == k]
        #tmp = measures[measures["Measure"] == k]["Service Number"].values
        #meterdf[k] = [int(x) in tmp for x in meterdf["Service Number"].values]
    meterdf.reset_index(inplace=True)
    

def select_data(df, **kwargs):
    df = df[[type(x) == int for x in df["Account"]]]
    df["Service Number"] = df["Account"] // 100
    for k, v in kwargs.items():
        if isinstance(v, str):
            select = df[k].str.upper() == v
        else:
            select = df[k] == v
        df = df[select]
    return df


if __name__ == "__main__":
    get_meter_data()
