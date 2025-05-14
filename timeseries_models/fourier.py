import numpy as np
import pandas as pd
import sys
sys.path.append("../")
import ami_data
import device_data
import matplotlib.pyplot as plt
import plot
import clean_data

def run():
    df = pd.read_parquet("test_sample100.parquet")
    clean_data.clean_all(df)
    clean_data.remove_all_devices(df)
    #meterdf, keys = device_data.get_meter_data()
    meterdf = pd.read_parquet("quick_meter100.parquet")
    df = ami_data.filter_key(df, meterdf, "solar")
    df = df.astype(float)
    iszero = df.sum(axis=0) == 0
    df.drop(columns=iszero[iszero].index.values, inplace=True)

    pca(df)

def plot_means(df):
    means = df.values.mean(axis=0)
    rms = np.linalg.norm(df.values, axis=0) / len(df)**.5
    inds = np.argsort(means)#[::-1]

    x = np.linspace(0, 1, len(means))
    plt.plot(means[inds], x, label="mean")
    plt.legend()
    plt.show()

def pca(df):
    data = df.values
    data -= data.mean(axis=1, keepdims=True)
    #data /= data.std(axis=1, keepdims=True)
    print(data.shape, data.dtype)
    corr = np.corrcoef(data, rowvar=False)
    vals, vecs = np.linalg.eigh(corr)
    print(vals.shape, vecs.shape)
    #plt.semilogy(vals)

    og_comp = data @ vecs
    og_comp /= np.linalg.norm(og_comp, axis=0)
    proj = og_comp.T @ data
    proj = proj[::-1]

    plt.scatter(proj[0], proj[1])
    plt.figure()

    n = 8 # how many to plot
    components = data @ vecs[:, :-n:-1]

    signs = np.sign(components.mean(axis=0))
    components *= signs
    v = np.around(vals[:-n:-1], 2)
    plot.avg_day(components, label=v)
    plt.legend()
    plt.show()
    

def plot_1dfft(df):
    ft = real_fft(df.values)
    x = np.amax(np.abs(ft), axis=-1)
    a = np.mean(np.abs(ft), axis=-1)
    inds = np.argsort(a*x)[::-1]
    gauss = (np.random.randn(len(df), 5) + 2) * 10
    gx = np.amax(np.abs(gauss), axis=1)
    ga = np.mean(np.abs(gauss), axis=1)
    ginds = np.argsort(ga*gx)[::-1]
    
    plt.semilogy(x[inds], label="max")
    plt.semilogy(a[inds], label="mean")
    plt.semilogy(gx[ginds], label="gmax")
    plt.semilogy(ga[ginds], label="gmean")
    plt.show()

def phase_fft(x, axis=None):
    if axis is None:
        axis = np.arange(len(x.shape) - 1)
    for ax in axis:
        y = np.swapaxes(x, ax, -1)
        n = y.shape[-1]
        m = n//2 + 1
        z = np.fft.fft(y, axis=-1)
        y[..., :m] = np.abs(z[..., :m])
        y[..., m:] = np.angle(z[..., 1:n-m+1])
        x = np.swapaxes(y, ax, -1)
    return x

def phase_ifft(x, axis=None):
    if axis is None:
        axis = np.arange(len(x.shape) - 1)
    for ax in axis:
        y = np.swapaxes(x, ax, -1)
        n = y.shape[-1]
        m = n//2 + 1
        z = np.zeros(y.shape, dtype=complex)
        z[..., :m] = y[..., :m]
        z[..., 1:n-m+1] *= np.exp(y[..., m:] * 1j)
        z[..., -1:-m:-1] = z[..., 1:m].conj()
        y = np.fft.ifft(z, axis=-1).real
        x = np.swapaxes(y, ax, -1)
    return x
    
def real_fft(x, axis=None):
    if axis is None:
        axis = np.arange(len(x.shape) - 1)
    for ax in axis:
        y = np.swapaxes(x, ax, -1)
        n = y.shape[-1]
        m = n//2 + 1
        z = np.fft.fft(y, axis=-1)
        y[..., :m] = z[..., :m].real
        y[..., m:] = z[..., 1:n-m+1].imag
        x = np.swapaxes(y, ax, -1)
    return x

def real_ifft(x, axis=None):
    if axis is None:
        axis = np.arange(len(x.shape) - 1)
    for ax in axis:
        y = np.swapaxes(x, ax, -1)
        n = y.shape[-1]
        m = n//2 + 1
        z = np.zeros(y.shape, dtype=complex)
        z[..., :m] = y[..., :m]
        z[..., 1:n-m+1] += y[..., m:] * 1j
        z[..., -1:-m:-1] = z[..., 1:m].conj()
        y = np.fft.ifft(z, axis=-1).real
        x = np.swapaxes(y, ax, -1)
    return x
    

if __name__ == "__main__":
    run()
