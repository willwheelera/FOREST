import pandas as pd
import numpy as np
import sys
sys.path.append("../MAPLE_BST_Demo")
from AMI_Player_Tools.parse_ami_data import format_ami_data
import device_data
import time
import matplotlib.pyplot as plt


def run():

    df = pd.read_parquet(f"data/Alburgh/home_charger_loads.parquet")
    df = avg_month(df)
    df = remove_gens(df)
    avgev = pd.DataFrame(df.pop("nocharger"))
    avgev["charger"] = df.mean(axis=1)
    avgev["diff"] = avgev["charger"] - avgev["nocharger"]

    fft = np.fft.fft(avgev.values.astype(float), axis=0)
    avgev.values[:] = fft_to_real(fft)
    fft = np.fft.fft(df.values.astype(float), axis=0)
    df.values[:] = fft_to_real(fft)
    plot_charger_load(df, avgev, save=False)

def plot_charger_load(df, avgev, save=True):
    df.plot(ls="-", lw=.5, legend=False)
    #plt.xticks(np.arange(12)*24, np.arange(12)+1)
    plt.title("Chargers each month")
    if save:
        plt.savefig("figures/home_charger_loads.pdf", bbox_inches="tight")

    avgev.plot()
    plt.title("Avg load w(/o) home charger")
    #plt.xticks(np.arange(12)*24, np.arange(12)+1)
    plt.legend(bbox_to_anchor=(1, 1))
    if save:
        plt.savefig("figures/home_charger_avg.pdf", bbox_inches="tight")
    plt.show()

def show_device_adoption():
    meterdf = pd.read_parquet("data/Alburgh/meters_devices_28.parquet")
    meterdf = meterdf[meterdf["home charger"]]
    keys = ["Meter Number", "cchp", "hpwh", "battery", "aev", "phev"]
    print(meterdf.columns)
    print(meterdf[keys])


def remove_gens(df):
    minload = df.min(axis=0)
    isgen = minload < 0
    print("generating")
    print(isgen)
    notgens = df.columns[~isgen]
    return df[notgens]


def avg_month(df):
    def _f(i):  
        return (i.month, i.hour)

    avg = df.groupby(_f).mean()
    return avg
    

def save_avg_nodevice(device="home charger"):
    t0 = time.perf_counter()
    df = pd.read_parquet("data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet", engine="pyarrow")
    printtime("loaded loads", t0)

    #meterdf, keys = device_data.get_meter_data()
    #meterdf = meterdf[meterdf["Substation"]=='28'].copy()
    meterdf = pd.read_parquet("data/Alburgh/meters_devices_28.parquet")
    printtime("loaded meters", t0)

    has_charger = meterdf[device]
    meternums = meterdf[has_charger]["Meter Number"].values.astype(int)
    is_charger = df.columns.astype(int).isin(meternums)
    avg = df[df.columns[~is_charger]].mean(axis=1)
    printtime("avg", t0)
    print(avg)

    df = df[df.columns[is_charger]]
    df["nocharger"] = avg
    df.to_parquet(f"data/Alburgh/{device}_loads.parquet")
    printtime(f"saved {device}", t0)


def printtime(s, t0): 
    print(s, round(time.perf_counter()-t0, 2), flush=True)

if __name__ == "__main__":
    run()
    
