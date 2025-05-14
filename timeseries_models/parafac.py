import pandas as pd
import numpy as np
import scipy.stats
import sys
sys.path.append("../")
from device_data import get_meter_data
import tensorly as tl
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def run():
    # === Load your data ===
    #fname = "../data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    #timeseries_df = pd.read_parquet(fname)[:-1]  # index: 'timestamp', columns: 'asset_id', data: 'value'
    #timeseries_df[timeseries_df.columns[:1000]].to_parquet("test_sample1000.parquet")
    timeseries_df = pd.read_parquet("test_sample100.parquet")

    data = timeseries_df.values.astype(float).reshape(365, 24, -1)
    data = data[:, :, :20] # just do a small sample

    mean = data.mean(axis=-1, keepdims=True)
    # which meters are furthest from mean?
    dist_from_mean = np.linalg.norm(data - mean, axis=(0, 1))
    outliers = np.argsort(dist_from_mean)[::-1]
    #plt.plot(data.mean(axis=0)[:, outliers[:5]])   
    #plt.show()
    #quit()
    data = data[:, :, outliers[5:]]

    # Don't actually need devices to do initial fit/reduction
    # === Match and align device flags ===
    #device_flags_df = get_meter_data()  # contains 'Meter Number', 'cchp', 'home charger', 'solar'
    #device_flags_df = device_flags_df.rename(columns={'Meter Number': 'asset_id'})
    #device_flags_df = device_flags_df.set_index('asset_id').loc[pivoted.index]
    #features = device_flags_df[['cchp', 'home charger', 'solar']].astype(float)

    data[69, 2] = (data[69, 1]+data[69, 3]) / 2 # error in data
    data[71, 6] = (data[71, 5]+data[71, 7]) / 2 # error in data
    #data = data / np.abs(np.amax(data, axis=-1, keepdims=True))
    #data = data / np.abs(np.std(data, axis=-1, keepdims=True))
    datafft = real_fft(data.copy())

    weights, factors = parafac(datafft, rank=7, n_iter_max=10, init='svd', normalize_factors=False)
    D, H, S = factors
    vals = np.einsum("i,di,hi,ni->dhn", weights, D, H, S)

    vals = real_ifft(vals)
    error = vals-data
    normerr = np.linalg.norm(error, axis=(0, 1))
    print("error", normerr)
    

    vals = vals.mean(axis=0)
    data = data.mean(axis=0)
    plt.plot(vals, ls=":")
    plt.plot(data)
    plt.figure()
    plt.plot(data - vals)

    skew = scipy.stats.skew(error, axis=-1, bias=False)
    kurt = scipy.stats.kurtosis(error, axis=-1, bias=False)
    fig, axs = plt.subplots(2, 1)
    b = 4
    axs[0].imshow(skew.T, vmin=-b, vmax=b, cmap="RdBu")
    axs[1].imshow(kurt.T, vmin=-b, vmax=b, cmap="RdBu")

    fig, axs = plt.subplots(4, 6)
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
    

def real_fft(x):
    for axis in (0, 1):
        y = np.swapaxes(x, axis, -1)
        n = y.shape[-1]
        m = n//2 + 1
        z = np.fft.fft(y, axis=-1)
        y[..., :m] = z[..., :m].real
        y[..., m:] = z[..., 1:n-m+1].imag
        x = np.swapaxes(y, axis, -1)
    return x

def real_ifft(x):
    for axis in (0, 1):
        y = np.swapaxes(x, axis, -1)
        n = y.shape[-1]
        m = n//2 + 1
        z = np.zeros(y.shape, dtype=complex)
        z[..., :m] = y[..., :m]
        z[..., 1:n-m+1] += y[..., m:] * 1j
        z[..., -1:-m:-1] = z[..., 1:m].conj()
        y = np.fft.ifft(z, axis=-1).real
        x = np.swapaxes(y, axis, -1)
    return x
    

if __name__ == "__main__":
    run()
