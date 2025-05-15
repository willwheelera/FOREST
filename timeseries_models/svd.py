import pandas as pd
import numpy as np
import scipy.stats
import sys
sys.path.append("../")
from device_data import get_meter_data
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import clean_data
import fourier
import plot


def run():
    # === Load your data ===
    #fname = "../data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    #df = pd.read_parquet(fname)[:-1]  # index: 'timestamp', columns: 'asset_id', data: 'value'
    #df[df.columns[:1000]].to_parquet("test_sample1000.parquet")
    df = pd.read_parquet("test_sample100.parquet")
    df = clean_data.clean_all(df)
    df = clean_data.remove_all_devices(df)

    data = df.values
    data = data[:, :20] # just do a small sample

    mean = data.mean(axis=-1, keepdims=True)
    # which meters are furthest from mean?
    dist_from_mean = np.linalg.norm(data - mean, axis=(0))
    outliers = np.argsort(dist_from_mean)[::-1]
    data = data[:, outliers[5:]]

    datafft = fourier.real_fft(data.copy())

    U, w, Vh = np.linalg.svd(datafft)
    rank = 7
    U = U[:, :rank]
    w = w[:rank]
    Vh = Vh[:rank]
    vals = np.einsum("k,mk,kn->mn", w, U, Vh)

    vals = fourier.real_ifft(vals)
    error = vals-data
    normerr = np.linalg.norm(error, axis=(0, 1))
    print("error", normerr)
    

    plot.avg_day(vals, ls=":")
    plot.avg_day(data)
    plt.figure()
    plot.avg_day(data - vals)

    skew = scipy.stats.skew(error, axis=-1, bias=False)
    kurt = scipy.stats.kurtosis(error, axis=-1, bias=False)
    fig, axs = plt.subplots(2, 1)
    b = 4
    axs[0].imshow(skew[:8736].reshape(52, 168, -1), vmin=-b, vmax=b, cmap="RdBu")
    axs[1].imshow(kurt[:8736].reshape(52, 168, -1), vmin=-b, vmax=b, cmap="RdBu")

    fig, axs = plt.subplots(4, 6)
    error = error.reshape(365, 24, -1)
    for i, ax in enumerate(axs.ravel()):
        sm.qqplot(error[:, i].ravel(), line="45", ax=ax)
    #plt.plot(vals-data)
    plt.show()
    
def error_df(x):
    inds = np.indices(x.shape).reshape(len(x.shape), -1)
    print("inds")
    print(inds.shape)
    print(x.shape)
    d = dict(day=inds[0], hour=inds[1], meter=inds[2], err=x.ravel())
    return pd.DataFrame(d)

def snsplot(errdf):
    fig, axs = plt.subplots(1, 3)
    keys = ["hour", "day", "meter"]
    for i, ax in enumerate(axs):
        sns.boxplot(data=errdf, x=keys[i], y="err", ax=ax)
    

if __name__ == "__main__":
    run()
