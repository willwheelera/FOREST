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

GROWTH="HIGH"
# General outline

def run_instance(nyears=20, year0=2025, seed=(1,)):
    """
    Ldata: past (2024) one-year dataframe, (hours, meters)
    meterdf: dataframe with meter numbers and device adoption status
    solardf: dataframe with meter numbers and solar status
    mapfile: information about which meters connect to which transformers
    """
    fulltimer = Timer()
    np.random.seed(seed)
    timer = Timer(30)
    timer.print("Starting")
    # Read in load data -- may want to use raw data instead
    fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    #mapfile = "data/Alburgh/transformer_load_map.json" # map meters to transformers
    mapfile = "data/Alburgh/transformer_map.pkl" # map meters to transformers

    Ldata, meterdf = read_in_data.load_meter_data(fname)
    m2t_map, TF_RATINGS = read_in_data.load_transformer_data(mapfile, Ldata.columns)
    m2t_frac = (m2t_map / TF_RATINGS)
    age_offset = np.zeros(m2t_map.shape[1])

    solardf = meterdf # todo.get_solar_data()
    nmeters = Ldata.shape[1]
    Hsize, Ssize = np.zeros(nmeters), np.zeros(nmeters)
    Eparams = np.zeros((4, 0))

    age_curves = []
    load_curves = []

    timer.print("data loaded")#, time.perf_counter() - t0)
    T0 = 110
    for year in np.arange(nyears) + year0:
        timer.reset()#t0 = time.perf_counter()
        timer.print(f"Reset timer {year}")
        H_g = placeholders.growth_rate_heatpumps(year, GROWTH)
        E_g = placeholders.growth_rate_evs(year, GROWTH)
        S_g = placeholders.growth_rate_solar(year, GROWTH)

        adoptH = placeholders.adopt_heatpumps(Ldata, meterdf, H_g)
        adoptE = placeholders.adopt_evs(Ldata, meterdf, E_g)
        adoptS = placeholders.adopt_solar(Ldata, solardf, S_g)
        Hsize += placeholders.size_heatpumps(Ldata, adoptH).values
        Ssize += -placeholders.size_solar(Ldata, adoptS).values
        newparams = ev_charging_model.generate_parameters(np.where(adoptE)[0])
        Eparams = np.concatenate([Eparams, newparams], axis=1)

        weather = weather_data.load_data.generate()

        LH = placeholders.generate_heatpump_load_profile(weather)[:, np.newaxis] # just one profile
        LE = ev_charging_model.generate_ev_load_profile(Eparams, len(adoptE))
        LS = sun_model.generate()[:, np.newaxis] # just one profile
        L0 = placeholders.generate_background_profile(Ldata)
        L = Hsize*LH + LE + Ssize*LS + L0
        pfactor = 1 - 0.1 * (~meterdf["solar"]).astype(float) # assume PF is 1 with inverter
        L = L / pfactor.values # power factor
        timer.print(f"meter loads calculated")#, time.perf_counter() - t0)

        # Transformer loads
        #L_tr = loads_to_transformers.meters_to_transformers_tuples(load_map, L).values
        #L_tr = m2t_mapper.meters_to_transformers(L)
        L_tr = L.values @ m2t_frac.values
        load_curves.append(L_tr)
        timer.print(f"transformer loads calculated")#, time.perf_counter() - t0)

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
    full_load_curve = np.concatenate(load_curves, axis=0)
    df = pd.DataFrame(data=full_load_curve, columns=m2t_map.columns)
    df.to_parquet(f"output/alburgh_tf_load_{tag}.parquet")

    fulltimer.print("Completed loads + failure curves")

    #meterdf.to_parquet(f"output/alburgh_meterdf_{seed}.parquet")
    #m2t_map.to_parquet(f"output/alburgh_m2t.parquet")

    Esize = np.zeros(len(Hsize))
    Esize[Eparams[0].astype(int)] = Eparams[1]
    sizes = pd.DataFrame(dict(H=Hsize, E=Esize, S=Ssize), index=m2t_map.index)
    tmp = m2t_map.T @ meterdf.loc[m2t_map.index][["cchp", "home charger", "solar"]]
    tf_device_sizes = m2t_map.T @ sizes
    tf_device_sizes["hasH"] = tmp["cchp"]
    tf_device_sizes["hasE"] = tmp["home charger"]
    tf_device_sizes["hasS"] = tmp["solar"]
    tf_device_sizes["age"] = df.iloc[-1]
    tf_device_sizes.to_parquet(f"output/alburgh_tf_devices_{tag}.parquet")
    
if __name__ == "__main__":
    
    run_instance(nyears=20, year0=2555)
