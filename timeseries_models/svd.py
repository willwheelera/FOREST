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
    df = clean_data.clean_all(df)
    df = clean_data.remove_all_devices(df)
    data = df.values

    #data = data / np.abs(np.amax(data, axis=-1, keepdims=True))
    #data = data / np.abs(np.std(data, axis=-1, keepdims=True))
    datafft = fourier.real_fft(data.copy())

def plot_mean(v):
    mean = v.mean(axis=0) # avg over year
    df = pd.DataFrame({"mean": mean})
    sns.kdeplot(data=df, x="mean")
    plt.show()
    

if __name__ == "__main__":
    run()
