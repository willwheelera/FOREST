import pandas as pd
import numpy as np
import scipy.sparse
import time
import json
import pickle
import concurrent.futures
import multiprocessing as mp
import os

import weather_data.load_data
import placeholders
import ev_charging_model
import sun_model
import loads_to_transformers
import transformer_aging
import read_in_data
from timer import Timer

# General outline

def compute_transformer_loads(nyears=20, year0=2025, GROWTH="HIGH", seeds=(1,)):
    fulltimer = Timer()
    timer = Timer(30, mute=False)
    timer.print("Starting")
    # Read in load data -- may want to use raw data instead
    fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    Ldata, meterdf = read_in_data.load_meter_data(fname)
    mapfile = "data/Alburgh/transformer_map.pkl" # map meters to transformers
    m2t_map, TF_RATINGS = read_in_data.load_transformer_data(mapfile, Ldata.columns)
    m2t_frac = (m2t_map / TF_RATINGS)
    
    timer.print("data loaded")#, time.perf_counter() - t0)
    timer.reset()
    nworkers = 8#os.cpu_count() - 1
    print(f"nworkers  {nworkers}")

    def submit_seed(seed):
        res_to_meters = {}
        for meters in np.array_split(Ldata.columns, nworkers):
            res = executor.submit(
                _calculate_meter_loads_subset,
                Ldata[meters], 
                meterdf.loc[meters], 
                m2t_frac.loc[meters],
                nyears, 
                year0, 
                GROWTH,
                seed=seed,
            )
            res_to_meters[res] = meters
        return (res_to_meters, seed)

    _seeds = [x for x in seeds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
        # Submit tasks for all seeds at once
        seed_results = []
        seed_results.append(submit_seed(_seeds.pop(0)))
        for _i in range(len(seeds)):
            if len(_seeds) > 0:
                seed_results.append(submit_seed(_seeds.pop(0)))

        # Start collecting, allowing new tasks to start as it collects
        #for i, (res_to_meters, seed) in enumerate(seed_results):
            (res_to_meters, seed) = seed_results.pop(0)
            timer.reset()
            timer.print(f"main reset {seed}")
            full_transformer_load = pd.DataFrame(
                data=np.zeros((m2t_map.shape[1], nyears*8760)), 
                index=m2t_map.columns,
            )
            sizes = pd.DataFrame(
                data=np.zeros((m2t_map.shape[0], 3)), 
                columns=["H", "E", "S"], 
                index=m2t_map.index,
            )
            timer.print(f"seed {seed} init outputs")

            for res in concurrent.futures.as_completed(res_to_meters):
                meters = res_to_meters[res]
                tf_tmp, tf_cols, meterdf.loc[meters], sizes.loc[meters] = res.result()
                full_transformer_load.values[tf_cols] += tf_tmp
            timer.print(f"seed {seed} collected results")

            load_outname = f"output/alburgh_tf_load_{GROWTH}_{year0}_{nyears}years_{seed}.parquet"
            full_transformer_load = full_transformer_load.T
            # Don't exceed tf capacity in generation
            full_transformer_load = np.maximum(full_transformer_load, -1.)
            full_transformer_load.to_parquet(load_outname)
            device_outname = f"output/alburgh_tf_devices_{GROWTH}_{year0}_{nyears}years_{seed}.parquet"
            save_tf_devices(device_outname, meterdf, m2t_map, sizes)
            timer.print(f"seed {seed} saved results")

    fulltimer.print("Complete")

def save_tf_devices(fname, meterdf, m2t_map, sizes):
    tmp = m2t_map.T @ meterdf.loc[m2t_map.index][["cchp", "home charger", "solar"]]
    tf_device_sizes = m2t_map.T @ sizes
    tf_device_sizes["hasH"] = tmp["cchp"]
    tf_device_sizes["hasE"] = tmp["home charger"]
    tf_device_sizes["hasS"] = tmp["solar"]
    tf_device_sizes.to_parquet(fname)

def _calculate_meter_loads_subset(Ldata, meterdf, m2t_frac, nyears, year0, GROWTH, seed=1):
    np.random.seed(seed)
    meterdf = meterdf.copy()
    nmeters = Ldata.shape[1]
    weather = weather_data.load_data.generate()
    full_meter_load = np.zeros((nyears, 8760, nmeters))
    Hsize, Ssize = np.zeros(nmeters), np.zeros(nmeters)
    Eparams = np.zeros((4, 0))
    tf_nonzero = m2t_frac.sum(axis=0) > 0
    tf_cols = m2t_frac.columns[tf_nonzero]
    m2t_frac = m2t_frac[tf_cols].values
    L = np.zeros((8760, nmeters))
    for year in np.arange(nyears).astype(int) + year0:
        H_g = placeholders.growth_rate_heatpumps(year, GROWTH)
        E_g = placeholders.growth_rate_evs(year, GROWTH)
        S_g = placeholders.growth_rate_solar(year, GROWTH)

        # adopt is an (nmeters,) boolean array for *new* devices to add
        adoptH = placeholders.adopt_heatpumps(Ldata, meterdf, H_g)
        adoptE = placeholders.adopt_evs(Ldata, meterdf, E_g)
        adoptS = placeholders.adopt_solar(Ldata, meterdf, S_g)
        # size is an (nmeters,) array, includes all devices added so far
        Hsize += placeholders.size_heatpumps(Ldata, adoptH).values
        Ssize += -placeholders.size_solar(Ldata, adoptS).values
        newparams = ev_charging_model.generate_parameters(np.where(adoptE)[0])
        Eparams = np.concatenate([Eparams, newparams], axis=1)

        LH = placeholders.generate_heatpump_load_profile(weather)[:, np.newaxis] # just one profile
        LE = ev_charging_model.generate_ev_load_profile(Eparams, len(adoptE))
        LS = sun_model.generate()[:, np.newaxis] # just one profile
        L0 = Ldata#placeholders.generate_background_profile(Ldata)
        L = Hsize*LH + LE + Ssize*LS + L0.values
        pfactor = 1 - 0.1 * (~meterdf["solar"]).values.astype(float) # assume PF is 1 with inverter
        full_meter_load[year-year0] = L / pfactor # power factor

    full_tf_load = (full_meter_load @ m2t_frac).reshape(-1, m2t_frac.shape[1])
    Esize = np.zeros(len(Hsize))
    Esize[Eparams[0].astype(int)] = Eparams[1]
    return full_tf_load.T, tf_nonzero, meterdf, np.stack([Hsize, Esize, Ssize], axis=1)

    
def generate_failure_curves(nyears=20, year0=2025, GROWTH="HIGH", seeds=(1,), label="", transcale=1.):
    timer = Timer(8, mute=False)
    timer.print("Starting")
    weather = weather_data.load_data.generate()
    weather = np.tile(weather, nyears)
    T0 = 110
    N = len(seeds)
    tf_device_sizes = pd.read_parquet(f"output/alburgh_tf_devices_{label}{GROWTH}_{year0}_{nyears}years_1.parquet")
    failure_curves = pd.DataFrame(columns=np.arange(8760*nyears), index=tf_device_sizes.index, data=0.)
    #failure_curves = pd.DataFrame(index=np.arange(8760*nyears), columns=tf_device_sizes.index, data=0.)
    timer.print("data loaded")
    ttag = "" if transcale == 1. else "tran"+str(int(transcale*100))
    
    nworkers = 10#os.cpu_count() - 1
    tf_inds = np.array_split(np.arange(len(tf_device_sizes)).astype(int), nworkers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
        args = (tf_inds, weather, T0, executor, nworkers, transcale)
        tag0 = f"{label}{GROWTH}_{year0}_{nyears}years_{seeds[0]}"
        res_to_tfs = _submit_batch_failure_curve(tag0, *args)
        timer.print(f"seed {seeds[0]} submitted")
        for seed in seeds[1:]:
            tag1 = f"{label}{GROWTH}_{year0}_{nyears}years_{seed}"
            res_to_tfs_new = _submit_batch_failure_curve(tag1, *args)
            timer.print(f"seed {seed} submitted")

            _collect_batch(res_to_tfs, failure_curves, ttag+tag0)
            tag0 = tag1
            res_to_tfs = res_to_tfs_new
            timer.print(f"seed {seed-1} failure curve")
        _collect_batch(res_to_tfs, failure_curves, ttag+tag0)
        timer.print(f"seed {seeds[-1]} failure curve")
        failure_curves = failure_curves.T
        failure_curves /= N
        timer.print(f"failure curves collected")

    failure_curves.to_parquet(f"output/alburgh_tf_failure_curves_{ttag}{label}{GROWTH}_{year0}_{nyears}years.parquet")
    timer.print("data saved")

def _collect_batch(res_to_tfs, failure_curves, tag, aging_skip=24):
    cols = failure_curves.columns[::aging_skip]
    agingdf = pd.DataFrame(columns=cols, index=failure_curves.index, data=0.)
    for res in concurrent.futures.as_completed(res_to_tfs):
        tfs = res_to_tfs[res]
        pfail, aging = res.result()
        failure_curves.values[tfs] += pfail
        agingdf.values[tfs] = aging
    outname = f"output/alburgh_tf_aging_{tag}.parquet"
    agingdf.T.to_parquet(outname)

def _submit_batch_failure_curve(tag, tf_inds, weather, T0, executor, nworkers, tscale):
    res_to_tfs = {}
    df = pd.read_parquet(f"output/alburgh_tf_load_{tag}.parquet") / tscale
    for i, tfi in enumerate(tf_inds):
        tfs = df.columns[tfi]
        res = executor.submit(
            _failure_curve_worker, df[tfs].values, weather, T0
        )
        res_to_tfs[res] = tfi#tf_inds[i]
    return res_to_tfs

def _failure_curve_worker(loads, weather, T0, aging_skip=24):
    #t0 = time.perf_counter()
    hotspot = transformer_aging.temperature_equations(loads, weather, T0=T0)
    aging = transformer_aging.effective_aging(hotspot)
    p_fail = transformer_aging.failure_prob(aging, eta=112, beta=3.5)
    #print("  worker time", round(time.perf_counter() - t0, 2))
    return p_fail.T, aging[::aging_skip].T

def collect_tf_device_average(nyears=20, year0=2025, GROWTH="HIGH", seeds=(1,), label=""):
    N = len(seeds)
    basename = f"output/alburgh_tf_devices_{label}{GROWTH}_{year0}_{nyears}years"
    df = pd.read_parquet(f"{basename}_{seeds[0]}.parquet")
    for seed in seeds[1:]:
        tmp = pd.read_parquet(f"{basename}_{seed}.parquet")
        df += tmp
    df /= N
    df.to_parquet(f"{basename}_avg.parquet")
    print("device averages collected")


if __name__ == "__main__":
    import workflow
    seeds = (np.arange(100) + 1).astype(int)
    nyears = 20
    year0 = 2025
    label = ""
    for GROWTH in ["MED", "MH", "HIGH"]:
        #workflow.compute_transformer_loads(nyears=nyears, year0=year0, GROWTH=GROWTH, seeds=seeds, label=label, LOADSCALING=1.5)
        generate_failure_curves(nyears=nyears, year0=year0, GROWTH=GROWTH, seeds=seeds, label=label, transcale=1.5)
        #collect_tf_device_average(nyears=nyears, year0=year0, GROWTH=GROWTH, seeds=seeds, label=label)
