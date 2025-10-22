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
from workflow import _calculate_loads_seed

# General outline


def generate_failure_curves(nyears=20, year0=2025, GROWTH="HIGH", seeds=(1,), label="", LOADSCALING=1.0, transcale=1.):
    timer = Timer()
    timer.print("Starting")
    # Read in load data -- may want to use raw data instead
    fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    mapfile = "data/Alburgh/transformer_map.pkl" # map meters to transformers

    Ldata, meterdf = read_in_data.load_meter_data(fname)

    m2t_map, TF_RATINGS = read_in_data.load_transformer_data(mapfile, Ldata.columns)
    m2t_frac = (m2t_map / TF_RATINGS)
    weather = weather_data.load_data.generate()
    weather = np.tile(weather, nyears)

    timer.print("data loaded")

    ###
    T0 = 110
    N = len(seeds)
    failure_curves = pd.DataFrame(columns=np.arange(8760*nyears), index=m2t_map.columns, data=0.)
    #failure_curves = pd.DataFrame(index=np.arange(8760*nyears), columns=tf_device_sizes.index, data=0.)
    ttag = "" if transcale == 1. else "tran"+str(int(transcale*100))
    
    nworkers = 10#os.cpu_count() - 1
    tf_inds = np.array_split(np.arange(m2t_map.shape[1]).astype(int), nworkers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
        load_args = (Ldata, meterdf, m2t_frac, m2t_map, nyears, year0, GROWTH)
        load_res = executor.submit(
            _calculate_loads_seed, *load_args, seeds[0], label=label, LOADSCALING=LOADSCALING
        )
        df, tf_device_sizes = load_res.result()
        args = (tf_inds, weather, T0, executor, nworkers, transcale)
        timer.print(f"seed {seeds[0]} load submitted")
        for i, seed in enumerate(seeds[1:]):
            load_res = executor.submit(
                _calculate_loads_seed, *load_args, seed, label=label, LOADSCALING=LOADSCALING
            )
            res_to_tfs = _submit_batch_failure_curve(df, *args)
            timer.print(f"seed {seeds[i]} submitted")

            _collect_batch(res_to_tfs, failure_curves)
            timer.print(f"seed {seeds[i]} failure curve")
            df, tmp = load_res.result()
            tf_device_sizes += tmp
        res_to_tfs = _submit_batch_failure_curve(df, *args)
        _collect_batch(res_to_tfs, failure_curves)
        timer.print(f"seed {seeds[-1]} failure curve")
    tf_device_sizes /= len(seeds)
    basename = f"output/alburgh_tf_devices_{label}{GROWTH}_{year0}_{nyears}years"
    tf_device_sizes.to_parquet(f"{basename}_avg.parquet")
    failure_curves = failure_curves.T
    failure_curves /= N
    timer.print(f"failure curves collected")

    failure_curves.to_parquet(f"output/alburgh_tf_failure_curves_{ttag}{label}{GROWTH}_{year0}_{nyears}years_{seeds[0]}-{seeds[-1]}.parquet")
    timer.print("data saved")

def _collect_batch(res_to_tfs, failure_curves, aging_skip=24):
    cols = failure_curves.columns[::aging_skip]
    for res in concurrent.futures.as_completed(res_to_tfs):
        tfs = res_to_tfs[res]
        pfail = res.result()
        failure_curves.values[tfs] += pfail

def _submit_batch_failure_curve(df, tf_inds, weather, T0, executor, nworkers, tscale):
    res_to_tfs = {}
    df = df / tscale
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
    return p_fail.T#, aging[::aging_skip].T


if __name__ == "__main__":
    seeds = (np.arange(1000) + 1).astype(int)
    nyears = 20
    year0 = 2025
    label = ""
    for GROWTH in ["MH"]:#["MED", "MH", "HIGH"]:
        generate_failure_curves(nyears=nyears, year0=year0, GROWTH=GROWTH, seeds=seeds, label=label, LOADSCALING=1., transcale=1.)
        #collect_tf_device_average(nyears=nyears, year0=year0, GROWTH=GROWTH, seeds=seeds, label=label)
