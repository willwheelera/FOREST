import pandas as pd
import numpy as np
import sys
sys.path.append("../")
from device_data import get_meter_data
import matplotlib.pyplot as plt
import seaborn as sns
import fourier
import clean_data


def run():
    # === Load your data ===
    #fname = "../data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    #timeseries_df = pd.read_parquet(fname)[:-1]  # index: 'timestamp', columns: 'asset_id', data: 'value'
    #timeseries_df[timeseries_df.columns[:1000]].to_parquet("test_sample1000.parquet")
    df = pd.read_parquet("test_sample1000.parquet")
    meterdf, keys = get_meter_data()
    df = clean_data.clean_all(df, meterdf)
    df = clean_data.remove_all_devices(df, meterdf)
    data = df.values

    #data = data / np.abs(np.amax(data, axis=-1, keepdims=True))
    #data = data / np.abs(np.std(data, axis=-1, keepdims=True))
    plot_mean(data)

def plot_mean(v):
    mean = v.mean(axis=0) # avg over year
    hist, bins = np.histogram(mean, bins=100, density=True)
    ch = np.cumsum(hist)
    plt.plot(bins[:-1], ch * np.diff(bins))

    n = len(mean)
    print("len mean", n)
    m = np.sort(mean)
    smoothm = smooth(m, 15)
    plt.plot(m, np.linspace(0, 1, n))
    plt.plot(smoothm, np.linspace(0, 1, n))

    plt.figure()
    plt.plot(bins[:-1], smooth(hist))
    plt.title("distribution across meters")
    plt.xlabel("mean load over 2024")
    plt.savefig("mean_distribution.pdf", bbox_inches="tight")
    plt.show()

def smooth(x, n=3):
    # x is 1d array
    # n is number of elements to average
    # Pad back
    y = np.zeros(len(x)+n-1)
    y[:-n+1] = x
    y[-n+1:] = x[-1]
    window = np.ones(n) / n
    return np.convolve(y, window, mode="valid")
    

if __name__ == "__main__":
    run()
