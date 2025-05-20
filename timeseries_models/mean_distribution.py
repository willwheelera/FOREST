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
    df = pd.read_parquet("test_sample100.parquet")
    meterdf = None if len(df) == 100 else get_meter_data()[0]
    df = clean_data.clean_all(df, meterdf)
    df = clean_data.remove_all_devices(df, meterdf)
    data = df.values

    #data = data / np.abs(np.amax(data, axis=-1, keepdims=True))
    #data = data / np.abs(np.std(data, axis=-1, keepdims=True))
    plot_mean(data)

def plot_max(v):
    # divide by mean - if the load curves scale, this helps compare
    # if the load curves are different sources added together, it's bad
    # Do we want a vector space where vectors all have the same sum? Maybe the same length (norm)? Probably not.....
    # how do we tell?
    #   PCA is components added together. but scaled anyway
    #   how much does the max vary?
    m = v.mean(axis=0)
    v = v - m
    _max = np.amax(v, axis=0) # this is max over the year... not that useful
    plt.scatter(m, _max) # is there a correlation between mean and spikiness?
    plt.show()
    

def plot_mean(v):
    mean = v.sum(axis=0) / 1000 # total over year in MWh
    n = len(mean)
    hist, bins = np.histogram(mean, bins=n//5)
    hist = hist / n # fraction of the total
    ch = np.cumsum(hist)
    plt.plot(bins[:-1], ch )

    print("len mean", n)
    m = np.sort(mean)
    smoothm = smooth(m, 3)
    plt.plot(m, np.linspace(0, 1, n))
    plt.plot(smoothm, np.linspace(0, 1, n))

    plt.figure()
    x = (bins[1:] + bins[:-1])/2
    plt.bar(x, hist)
    plt.plot(x, smooth(hist))
    plt.title("distribution across meters")
    plt.xlabel("total load over 2024")
    plt.ylabel("fraction of meters")
    plt.savefig("total_distribution.pdf", bbox_inches="tight")
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
