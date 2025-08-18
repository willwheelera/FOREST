import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import weather_data.load_data
import os
import time
import numba

def run():
    path = "data/Alburgh/"
    tf_fname = "normalized_transformer_loading.parquet"
    
    # Read in transformer rated kVA
    ratings = pd.read_csv(path+"transformer_ratings.csv", index_col=0)
    newratings = pd.read_csv(path+"2025_transformer_ratings.csv", index_col=0)
    ratings = ratings.set_index("tf_name")
    newratings = newratings.set_index("gs_equipment_location")

    # Try to fix incorrect values
    bigkeys = ['E72805100040038', 'E72805103079841', 'E72805103080326',
   'E72805103080328', 'E72805103097973', 'E72805203078657',
   'E72805203078706', 'E72805203078859', 'E72805203078866',
   'E72805203079531', 'E72805203079711', 'E72805203080317']
    ratings.loc[bigkeys, "ratedKVA"] = newratings.loc[bigkeys, "gs_rated_kva"]
    ratings.loc["E72805203096474", "ratedKVA"] = 100

    # Read in transformer loads
    load_fname = path+"transformer_loads.parquet"
    load_tfdf = pd.read_parquet(load_fname)
    #gen_tfdf = pd.read_parquet(load_fname.replace("load", "gen"))
    tfdf = load_tfdf#.subtract(gen_tfdf, fill_value=0.) # not necessary, already processed to include gen
    ratings = ratings.loc[tfdf.columns]
    tfdf = (tfdf / ratings["ratedKVA"].values)[:8760] / 0.9 # power factor
    tfdf.to_parquet(path+tf_fname)

    # Read in weather data
    weather = weather_data.load_data.load_data()
    TMIN = (weather["TMIN"].values[:365] - 32) * 5/9
    TMAX = (weather["TMAX"].values[:365] - 32) * 5/9
    weather = weather_data.load_data.interpolate(weather)
    weather = (weather - 32) * 5/9
    
    # Solve for transformer temperature and aging 
    t0 = time.perf_counter()
    temps = temperature_equations(tfdf.values, weather)
    aging = effective_aging(temps) # in years
    age = aging[-1]
    ratings["age"] = age
    ratings["index"] = np.arange(len(ratings))
    print("time", time.perf_counter() - t0)
    big = np.where(age > 5)[0]
    bigcols = tfdf.columns[big]
    print(big)
    print(bigcols)

    selection = big
    print(age[selection])
    print(ratings.loc[bigcols])
    
    save=True
    plot_hotspot(temps[:, selection], TMIN, TMAX, label=bigcols, save=save)
    plot_loads(tfdf.values[:, selection], label=bigcols, save=save)
    plot_aging(aging[:, selection], label=bigcols, save=save)
    plot_age(age, save=save)

    plt.show()

def plot_hotspot(temps, TMIN, TMAX, label=None, save=True):
    plt.figure(figsize=(8, 3.0))
    plt.plot(temps, label=label) 
    plt.title("hot-spot temperature")
    plt.ylabel("Temperature (C)")
    plt.plot(np.arange(17, 8760, 24), TMAX)
    plt.plot(np.arange(5, 8760, 24), TMIN)
    plt.axhline(y=110, ls=":")
    plt.axhline(y=120, ls=":")
    plt.axhline(y=130, ls=":")
    plt.axhline(y=140, ls=":")
    plt.xlabel("month in 2024")
    plt.xticks(np.arange(12)*24*30, np.arange(12))
    plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left')
    plt.subplots_adjust(right=0.7)
    if save:
        plt.savefig("figures/hotspot_temperature.pdf", bbox_inches="tight")

def plot_loads(tfdf_vals, label=None, save=True):
    plt.figure(figsize=(8, 3.0))
    plt.plot(tfdf_vals, label=label)
    plt.axhline(y=1, ls=":")
    plt.axhline(y=-1, ls=":")
    plt.ylabel("loads / rated kVA")
    plt.title("transformer loading")
    plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.xlabel("month in 2024")
    plt.xticks(np.arange(12)*24*30, np.arange(12))
    if save:
        plt.savefig("figures/transformer_loads.pdf", bbox_inches="tight")

def plot_aging(aging, label=None, save=True):
    plt.figure(figsize=(8, 3.0))
    plt.plot(aging, label=label)
    plt.axhline(y=20.55, ls=":")
    plt.ylabel("Effective age (years)")
    plt.title("transformer aging")
    plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.xlabel("month in 2024")
    plt.xticks(np.arange(12)*24*30, np.arange(12))
    if save:
        plt.savefig("figures/transformer_aging.pdf", bbox_inches="tight")

def plot_age(age, save=True):
    plt.figure(figsize=(3, 3))
    plt.semilogy(np.sort(age)[::-1])
    plt.axhline(y=1)
    plt.xlabel("transformer index (sorted)")
    plt.title("aging over 1 year")
    plt.ylabel("# years aged")
    if save:
        plt.savefig("figures/transformer_ages.pdf", bbox_inches="tight")

    
TAU = 0.4090
ACOEFF = 23.277
BCOEFF = 0.5910
CCOEFF = 7.060
@numba.njit
def temperature_equations(loading, temperature, T0=110):
    """
    loading: transformer loading DataFrame, normalized by rated kVA, array(time, xfmr)
    temperature: ambient temperature, array(time) 
    T0: starting hottest-spot temperature of windings; scalar or array(xfmr)
    """
    # if you have avg(load) = L over an hour, what is avg(load^2)? L^2 + variance
    external = ACOEFF * loading**2 + BCOEFF * temperature[:, np.newaxis] + CCOEFF
    out = np.zeros_like(loading)
    T = T0 * TAU + external[0]
    for i, ext in enumerate(external):
        out[i] = T
        T = T * TAU + ext
    return out
    
def plot_aging_curve():
    fig, axs = plt.subplots(1, 2, figsize=(6,3))
    T = np.arange(200)
    FA = np.exp(-15000*(1/(T+273) - 1/383))
    plt.sca(axs[0])
    plt.semilogy(T, FA)
    plt.axhline(y=1, ls=":")
    plt.ylabel("aging acceleration factor")
    plt.xlabel("Temperature (C)")
    plt.title("aging curve")

    plt.sca(axs[1])
    plt.plot(T[:140], FA[:140])
    plt.axhline(y=1, ls=":")
    plt.text(10, 10, r"$e^{-15000K \left(\frac{1}{T+273K} - \frac{1}{383K}\right)}$", size="large")
    plt.tight_layout()
    plt.savefig("figures/aging_curve.pdf", bbox_inches="tight")
    plt.show()

def effective_aging(T):
    FA = np.exp(-15000*(1/(T+273) - 1/383))
    return FA.cumsum(axis=0) / 8760 # in years

# Martin et al, Investigation Into Modeling Australian Power Transformer Failure and Retirement Statistics
# IEEE TRANSACTIONS ON POWER DELIVERY, VOL. 33, NO. 4, AUGUST 2018
# Digital Object Identifier 10.1109/TPWRD.2018.2814588
@numba.njit
def failure_prob(age, eta=112, beta=3.5):
    return 1 - np.exp(-(age/eta)**beta)

def plot_failure_curve():
    t = np.linspace(0, 200, 100)
    f = failure_prob(t)
    plt.figure(figsize=(3,3))
    plt.plot(t, f)
    plt.xlabel("age (years)")
    plt.ylabel("failure probability")
    plt.savefig("figures/failure_probability_curve.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run()


