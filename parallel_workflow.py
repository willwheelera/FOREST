import pandas as pd
import numpy as np
import scipy.sparse
import time
import json
import pickle
import weather_data.load_data
import placeholders
import sun_model
import loads_to_transformers
import transformer_aging
import device_data
import read_in_data
from timer import Timer
import concurrent.futures
from numba import njit

# General outline

def compute_transformer_loads(nyears=20, year0=2025, seeds=(1,), executor=None, nworkers=1):
    timer = Timer(30, mute=False)
    timer.print("Starting")
    # Read in load data -- may want to use raw data instead
    fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    Ldata, meterdf = _load_meter_data(fname)
    mapfile = "data/Alburgh/transformer_map.pkl" # map meters to transformers
    m2t_map, TF_RATINGS = _load_transformer_data(mapfile, Ldata.columns)
    m2t_frac = (m2t_map / TF_RATINGS)
    
    timer.print("data loaded")#, time.perf_counter() - t0)

    for seed in seeds:
        #run_instance(Ldata, meterdf, m2t_map, TF_RATINGS, nyears, year0, seed=seed)
        #break

        # Meter loads
        res_list = []
        for meters in np.array_split(Ldata.columns, nworkers):
            res = executor.submit(
                _calculate_meter_loads_subset,
                Ldata[meters], 
                meterdf.loc[meters], 
                m2t_frac.loc[meters],
                nyears, 
                year0, 
                seed=seed,
            )
            res_list.append((res, meters))

        #full_meter_load = pd.DataFrame(
        #    data=np.zeros((m2t_map.shape[0], nyears*8760)), 
        #    index=m2t_map.index,
        #)
        full_transformer_load = pd.DataFrame(
            data=np.zeros((m2t_map.shape[1], nyears*8760)), 
            index=m2t_map.columns,
        )
        sizes = pd.DataFrame(
            data=np.zeros((Ldata.shape[1], 3)), 
            columns=["H", "E", "S"], 
            index=m2t_map.index,
        )
        #m2t_frac = m2t_map.values / TF_RATINGS
        timer.print("sent jobs and init arrays")
        for res, meters in res_list:
            t0 = Timer()
            #full_meter_load.loc[meters], meterdf.loc[meters], sizes.loc[meters] = res.result()
            tf_tmp, tf_cols, meterdf.loc[meters], sizes.loc[meters] = res.result()
            t0.print("  got result", end=" ")
            full_transformer_load.loc[tf_cols] += tf_tmp
            t0.print("  added")
            #m2t_frac = (m2t_map.loc[meters].values / TF_RATINGS)
            #full_transformer_load += meter_load @ m2t_frac
            #t0.print("  added transformer load")
        timer.print("collected results")

        #full_transformer_load = full_meter_load.T @ m2t_frac
        full_transformer_load = full_transformer_load.T
        timer.print("computed tf loads")
            
            
        load_outname = f"output/alburgh_tf_load_{year0}_{nyears}years_{seed}.parquet"
        full_transformer_load.to_parquet(load_outname)
        device_outname = f"output/alburgh_tf_devices_{year0}_{nyears}years_{seed}.parquet"
        save_tf_devices(device_outname, meterdf, m2t_map, sizes)
        timer.print("saved results")

            
def _load_meter_data(fname):
    Ldata = pd.read_parquet(fname)
    meterdf, keys = device_data.get_meter_data("sub28")
    #meterdf = meterdf[meterdf["Substation"] == 28]
    meterdf.set_index("Meter Number", inplace=True)
    Ldata = Ldata[Ldata.columns[Ldata.columns.isin(meterdf.index)]]
    Ldata = Ldata[:8760].astype(float)
    meterdf = meterdf.loc[Ldata.columns]
    return Ldata, meterdf

def _load_transformer_data(mapfile, columns):
    with open(mapfile, "rb") as f:
        load_map = pickle.load(f)["load_map"]
    #load_map = [l for l in load_map if l[0] in Ldata.columns]
    m2t_map = loads_to_transformers.meter_to_transformer_matrix(load_map, columns)
    tf_ratings = read_in_data.read_transformer_ratings("Alburgh")
    tf_ratings = tf_ratings.loc[m2t_map.columns]
    return m2t_map, tf_ratings["ratedKVA"].values

def run_instance(Ldata, meterdf, m2t_map, TF_RATINGS, nyears, year0, seed=1):
    """
    Ldata: past (2024) one-year dataframe, (hours, meters)
    meterdf: dataframe with meter numbers and device adoption status
    solardf: dataframe with meter numbers and solar status
    mapfile: information about which meters connect to which transformers
    """
    #seed = 1 #int(round(time.time() % 1e8, 1))
    np.random.seed(seed)
    timer = Timer(30, mute=False)

    meterdf = meterdf.copy()
    solardf = meterdf # todo.get_solar_data()
    Hsize, Esize, Ssize = np.zeros(Ldata.shape[1]), np.zeros(Ldata.shape[1]), np.zeros(Ldata.shape[1])

    m2t_frac = m2t_map.values / TF_RATINGS
    full_load_curve = np.zeros((nyears*8760, m2t_frac.shape[1]))
    for year in np.arange(nyears).astype(int) + year0:
        H_g = placeholders.growth_rate_heatpumps(year)
        E_g = placeholders.growth_rate_evs(year)
        S_g = placeholders.growth_rate_solar(year)

        # adopt is an (nmeters,) boolean array for *new* devices to add
        adoptH = placeholders.adopt_heatpumps(Ldata, meterdf, H_g)
        adoptE = placeholders.adopt_evs(Ldata, meterdf, E_g)
        adoptS = placeholders.adopt_solar(Ldata, solardf, S_g)
        # size is an (nmeters,) array, includes all devices added so far
        Hsize += placeholders.size_heatpumps(Ldata, adoptH).values
        Esize += placeholders.size_evs(Ldata, adoptE).values
        Ssize += -placeholders.size_solar(Ldata, adoptS).values

        weather = weather_data.load_data.generate()

        LH = placeholders.generate_heatpump_load_profile(weather)[:, np.newaxis] # just one profile
        LE = placeholders.generate_ev_load_profile(Esize)
        LS = sun_model.generate()[:, np.newaxis] # just one profile
        #L0 = placeholders.generate_background_profile(Ldata)
        #L = Hsize*LH + LE + Ssize*LS + L0
        #pfactor = 1 - 0.1 * (~meterdf["solar"]).astype(float) # assume PF is 1 with inverter
        #L = L / pfactor # power factor
        L0 = placeholders.generate_background_profile(Ldata)
        L0 += load_profiles(weather, Hsize, Esize, Ssize, (~meterdf["solar"].values).astype(float))
        L = L0
        timer.print(f"year {year} meter loads")

        # Transformer loads
        start = (year - year0) * 8760
        full_load_curve[start:start+8760] = L.values @ m2t_frac
        timer.print(f"year {year} transformer loads")

    df = pd.DataFrame(data=full_load_curve, columns=m2t_map.columns)
    df.to_parquet(f"output/alburgh_tf_load_{year0}_{nyears}years_{seed}.parquet")
    device_outname = f"output/alburgh_tf_devices_{year0}_{nyears}years_{seed}.parquet"
    sizes = pd.DataFrame(
        data=dict(H=Hsize, E=Esize, S=Ssize),
        index=m2t_map.index,
    )
    save_tf_devices(device_outname, meterdf, m2t_map, sizes)
    timer.print(f"seed {seed} saved transformer loads")

#@njit
def load_profiles(weather, Hsize, Esize, Ssize, LH, LE, LS, notsolar):
    #LH = placeholders.generate_heatpump_load_profile(weather)#[:, np.newaxis] # just one profile
    #LE = placeholders.generate_ev_load_profile(Esize)
    #LS = sun_model.generate()#[:, np.newaxis] # just one profile
    L = np.zeros((len(LH), len(Hsize)))
    pfactor = 1 - 0.1 * (notsolar) # assume PF is 1 with inverter
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            L[i, j] = (Hsize[j] * LH[i] + LE[i, j] + Ssize[j] * LS[i]) / pfactor[j]
    #L = Hsize*LH + LE + Ssize*LS
    return L #/ pfactor # power factor

def save_tf_devices(fname, meterdf, m2t_map, sizes):
    tmp = m2t_map.T @ meterdf.loc[m2t_map.index][["cchp", "home charger", "solar"]]
    tf_device_sizes = m2t_map.T @ sizes
    tf_device_sizes["hasH"] = tmp["cchp"]
    tf_device_sizes["hasE"] = tmp["home charger"]
    tf_device_sizes["hasS"] = tmp["solar"]
    tf_device_sizes.to_parquet(fname)

def _calculate_meter_loads_subset(Ldata, meterdf, m2t_frac, nyears, year0, seed=1):
    nmeters = Ldata.shape[1]
    weather = weather_data.load_data.generate()
    full_meter_load = np.zeros((nyears*8760, nmeters))
    Hsize, Esize, Ssize = np.zeros(nmeters), np.zeros(nmeters), np.zeros(nmeters)
    tf_nonzero = m2t_frac.sum(axis=0) > 0
    tf_cols = m2t_frac.columns[tf_nonzero]
    m2t_frac = m2t_frac[tf_cols].values
    L = np.zeros((8760, nmeters))
    for year in np.arange(nyears).astype(int) + year0:
        H_g = placeholders.growth_rate_heatpumps(year)
        E_g = placeholders.growth_rate_evs(year)
        S_g = placeholders.growth_rate_solar(year)

        # adopt is an (nmeters,) boolean array for *new* devices to add
        adoptH = placeholders.adopt_heatpumps(Ldata, meterdf, H_g)
        adoptE = placeholders.adopt_evs(Ldata, meterdf, E_g)
        adoptS = placeholders.adopt_solar(Ldata, meterdf, S_g)
        # size is an (nmeters,) array, includes all devices added so far
        Hsize += placeholders.size_heatpumps(Ldata, adoptH).values
        Esize += placeholders.size_evs(Ldata, adoptE).values
        Ssize += -placeholders.size_solar(Ldata, adoptS).values

        LH = placeholders.generate_heatpump_load_profile(weather)[:, np.newaxis] # just one profile
        LE = placeholders.generate_ev_load_profile(Esize)
        LS = sun_model.generate()[:, np.newaxis] # just one profile
        L0 = Ldata#placeholders.generate_background_profile(Ldata)
        L[:] = Hsize*LH + LE + Ssize*LS + L0.values
        pfactor = 1 - 0.1 * (~meterdf["solar"]).values.astype(float) # assume PF is 1 with inverter
        #L0 = placeholders.generate_background_profile(Ldata)
        #L0 += load_profiles(weather, Hsize, Esize, Ssize, (~meterdf["solar"].values).astype(float))
        #L = L0
        start = (year - year0) * 8760
        full_meter_load[start:start+8760] = L / pfactor # power factor
    full_tf_load = full_meter_load @ m2t_frac
    return full_tf_load.T, tf_cols, meterdf, np.stack([Hsize, Esize, Ssize], axis=1)

    
def generate_failure_curves(nyears=20, year0=2025, seeds=(1,)):
    timer = Timer(8, mute=False)
    timer.print("Starting")
    weather = weather_data.load_data.generate()
    weather = np.tile(weather, nyears)
    T0 = 110
    N = len(seeds)
    tf_device_sizes = pd.read_parquet(f"output/alburgh_tf_devices_{year0}_{nyears}years_1.parquet")
    failure_curves = pd.DataFrame(index=np.arange(8760*nyears), columns=tf_device_sizes.index, data=0.)
    timer.print("data loaded")
    
    for seed in seeds:
        timer.reset()
        df = pd.read_parquet(f"output/alburgh_tf_load_{year0}_{nyears}years_{seed}.parquet")
        hotspot = transformer_aging.temperature_equations(df.values, weather, T0=T0)
        aging = transformer_aging.effective_aging(hotspot)
        p_fail = transformer_aging.failure_prob(aging, eta=112, beta=3.5)
        failure_curves += p_fail / N
        timer.print(f"seed {seed} failure curve")

    failure_curves.to_parquet(f"output/alburgh_tf_failure_curves_{year0}_{nyears}years.parquet")

def collect_tf_device_average(nyears=20, year0=2025, seeds=(1,)):
    N = len(seeds)
    basename = f"output/alburgh_tf_devices_{year0}_{nyears}years"
    df = pd.read_parquet(f"{basename}_1.parquet")
    for seed in seeds[1:]:
        df += pd.read_parquet(f"{basename}_{seed}.parquet")
    df /= N
    df.to_parquet(f"{basename}_avg.parquet")


def compute_transformer_loads_parallel(nyears=0, year0=0, seeds=(1,)):
    timer = Timer()
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        nworkers = executor._max_workers
        print(f"Running on {nworkers} processes")
        results = []
        for subset in np.array_split(seeds, nworkers):
            r = executor.submit(compute_transformer_loads, nyears=nyears, year0=year0, seeds=subset)
            results.append(r)
        for r in results:
            r.result()
    # Results saved to disk, no post-processing to be done
    timer.print("Finished compute_transformer_loads")

def generate_failure_curves_parallel(nyears=0, year0=0, seeds=(1,)):
    timer = Timer()
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        nworkers = executor._max_workers
        print(f"Running on {nworkers} processes")
        results = []
        for subset in np.array_split(seeds, nworkers):
            r = executor.submit(generate_failure_curves, nyears=nyears, year0=year0, seeds=subset)
            results.append(r)
        for r in results:
            r.result()
    # Results saved to disk, no post-processing to be done
    timer.print("Finished generate_failure_curves")


if __name__ == "__main__":
    seeds = (np.arange(2) + 1).astype(int)
    nyears = 20
    year0 = 4040
    seeds = [1]
    with concurrent.futures.ProcessPoolExecutor(max_workers=19) as executor:
        compute_transformer_loads(
            nyears=nyears, 
            year0=year0, 
            seeds=seeds, 
            executor=executor, 
            nworkers=executor._max_workers,
        )
    #generate_failure_curves_parallel(nyears=nyears, year0=year0, seeds=seeds)
    #collect_tf_device_average(nyears=nyears, year0=year0, seeds=seeds)
