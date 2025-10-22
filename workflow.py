import pandas as pd
import numpy as np
import scipy.sparse
import time
import json
import pickle

import weather_data.load_data
import placeholders
import sun_model
import ev_charging_model
import loads_to_transformers
import transformer_aging
import device_data
import read_in_data
from timer import Timer



def compute_transformer_loads(nyears=20, year0=2025, GROWTH="HIGH", seeds=(1,), label="", LOADSCALING=1.0):
    """
    Ldata: past (2024) one-year dataframe, (hours, meters)
    meterdf: dataframe with meter numbers and device adoption status
    solardf: dataframe with meter numbers and solar status
    mapfile: information about which meters connect to which transformers
    """
    fulltimer = Timer()
    fulltimer.print("Starting")
    # Read in load data -- may want to use raw data instead
    fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    mapfile = "data/Alburgh/transformer_map.pkl" # map meters to transformers

    Ldata, meterdf = read_in_data.load_meter_data(fname)

    m2t_map, TF_RATINGS = read_in_data.load_transformer_data(mapfile, Ldata.columns)
    m2t_frac = (m2t_map / TF_RATINGS)

    fulltimer.print("data loaded")

    for seed in seeds:
        _calculate_loads_seed(Ldata, meterdf, m2t_frac, m2t_map, nyears, year0, GROWTH, seed, label, LOADSCALING)

    fulltimer.print("Completed loads + failure curves")


def _calculate_loads_seed(Ldata, meterdf, m2t_frac, m2t_map, nyears, year0, GROWTH, seed, label="", LOADSCALING=1.):
    timer = Timer(12)
    timer.print(f"seed {seed}", end="    ")#, time.perf_counter() - t0)
    np.random.seed(seed)
    meterdf = meterdf.copy()
    nmeters = Ldata.shape[1]
    weather = weather_data.load_data.generate()
    ntfs = m2t_frac.shape[1]
    full_tf_load = np.zeros((nyears, 8760, ntfs))
    Hsize, Ssize = np.zeros(nmeters), np.zeros(nmeters)
    Eparams = np.zeros((4, 0))

    T0 = 110
    for year in np.arange(nyears) + year0:
        H_g = placeholders.growth_rate_heatpumps(year, GROWTH)
        E_g = placeholders.growth_rate_evs(year, GROWTH)
        S_g = placeholders.growth_rate_solar(year, GROWTH)

        adoptH = placeholders.adopt_heatpumps(Ldata, meterdf, H_g)
        adoptE = placeholders.adopt_evs(Ldata, meterdf, E_g)
        adoptS = placeholders.adopt_solar(Ldata, meterdf, S_g)
        Hsize += placeholders.size_heatpumps(Ldata, adoptH).values
        Ssize += -placeholders.size_solar(Ldata, adoptS).values
        newparams = ev_charging_model.generate_parameters(np.where(adoptE)[0])
        Eparams = np.concatenate([Eparams, newparams], axis=1)

        LH = placeholders.generate_heatpump_load_profile(weather)[:, np.newaxis] # just one profile
        LE = ev_charging_model.generate_ev_load_profile(Eparams, len(adoptE))
        LS = sun_model.generate()[:, np.newaxis] # just one profile
        # higher base load
        #L0 = LOADSCALING * placeholders.generate_background_profile(Ldata)
        L = Hsize*LH + LE + Ssize*LS + LOADSCALING*Ldata
        pfactor = 1 - 0.1 * (~meterdf["solar"]).astype(float) # assume PF is 1 with inverter
        L = L / pfactor.values # power factor

        # Transformer loads
        #L_tr = loads_to_transformers.meters_to_transformers_tuples(load_map, L).values
        #L_tr = m2t_mapper.meters_to_transformers(L)
        full_tf_load[year-year0] = L.values @ m2t_frac.values

    timer.print(f"{seed} calculated", end="    ")

    df = pd.DataFrame(data=full_tf_load.reshape(-1, ntfs), columns=m2t_map.columns)
    tag = f"{label}{GROWTH}_{year0}_{nyears}years_{seed}"
    #df.to_parquet(f"output/alburgh_tf_load_{tag}.parquet")

    Esize = np.zeros(len(Hsize))
    Esize[Eparams[0].astype(int)] = Eparams[1]
    sizes = pd.DataFrame(dict(H=Hsize, E=Esize, S=Ssize), index=m2t_map.index)
    tmp = m2t_map.T @ meterdf.loc[m2t_map.index][["cchp", "home charger", "solar"]]
    tf_device_sizes = m2t_map.T @ sizes
    tf_device_sizes["hasH"] = tmp["cchp"]
    tf_device_sizes["hasE"] = tmp["home charger"]
    tf_device_sizes["hasS"] = tmp["solar"]
    tf_device_sizes["age"] = df.iloc[-1]
    #tf_device_sizes.to_parquet(f"output/alburgh_tf_devices_{tag}.parquet")
    
    timer.print(f"saved")

    return df, tf_device_sizes

def generate_failure_curves(nyears=20, year0=2025, seeds=(1,)):
    # TODO not implemented yet
    if True:
        # Transformer aging
        hotspot = transformer_aging.temperature_equations(L_tr, weather, T0=T0)
        T0 = hotspot[-1] # for iterating multiple years if desired
        aging = transformer_aging.effective_aging(hotspot)
        timer.print(f"aging calculated")#, time.perf_counter() - t0)
        aging += age_offset
        age_offset = aging[-1]
        age_curves.append(aging)
        timer.print(f"year {year} done")#, time.perf_counter() - t0)
        
    full_age_curve = np.concatenate(age_curves, axis=0)
    df = pd.DataFrame(data=full_age_curve, columns=m2t_map.columns)
    tag = f"{GROWTH}_{year0}_{nyears}years_{seed}"
    df.to_parquet(f"output/alburgh_tf_aging_{tag}.parquet")

    #meterdf.to_parquet(f"output/alburgh_meterdf_{seed}.parquet")
    #m2t_map.to_parquet(f"output/alburgh_m2t.parquet")

if __name__ == "__main__":
    seeds = (np.arange(100) + 1).astype(int)
    nyears = 20
    year0 = 2025
    GROWTH = "MH"
    compute_transformer_loads(nyears=nyears, year0=year0, GROWTH=GROWTH, seeds=seeds)
