import pandas as pd
import numpy as np
import scipy
import time
import json
import pickle
import weather_data.load_data
import placeholders
import sun_model
import loads_to_transformers
import transformer_aging
import device_data


# General outline

def run_instance(nyears=20, year0=2025):
    """
    Ldata: past (2024) one-year dataframe, (hours, meters)
    meterdf: dataframe with meter numbers and device adoption status
    solardf: dataframe with meter numbers and solar status
    mapfile: information about which meters connect to which transformers
    """
    t0 = time.perf_counter()
    print("Starting")
    # Read in load data -- may want to use raw data instead
    fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    #mapfile = "data/Alburgh/transformer_load_map.json" # map meters to transformers
    mapfile = "data/Alburgh/transformer_map.pkl" # map meters to transformers
    Ldata = pd.read_parquet(fname)
    meterdf, keys = device_data.get_meter_data("sub28")
    #meterdf = meterdf[meterdf["Substation"] == 28]
    meterdf.set_index("Meter Number", inplace=True)
    Ldata = Ldata[Ldata.columns[Ldata.columns.isin(meterdf.index)]]
    Ldata = Ldata[:8760].astype(float)
    meterdf = meterdf.loc[Ldata.columns]

    solardf = meterdf # todo.get_solar_data()
    Hsize, Esize, Ssize = np.zeros(Ldata.shape[1]), np.zeros(Ldata.shape[1]), np.zeros(Ldata.shape[1])

    age_curves = []
    if mapfile.endswith("json"):
        with open(mapfile, "r") as f:
            load_map = json.load(f)
        age_offset = np.zeros(len(load_map.keys()))
    else:
        with open(mapfile, "rb") as f:
            load_map = pickle.load(f)["load_map"]
        load_map = [l for l in load_map if l[0] in Ldata.columns]
        age_offset = np.zeros(len(np.unique([m[1] for m in load_map])))
        # TODO try sparse matrix version # load_map_sparse = scipy.sparse.coo_array((np.ones(len(load_map)), np.array(load_map).T))

    print("data loaded", time.perf_counter() - t0)
    T0 = 110
    for year in np.arange(nyears) + year0:
        H_g = placeholders.growth_rate_heatpumps(year)
        E_g = placeholders.growth_rate_evs(year)
        S_g = placeholders.growth_rate_solar(year)
        print(f"{year} growth rates computed", time.perf_counter() - t0)

        adoptH = placeholders.adopt_heatpumps(Ldata, meterdf, H_g)
        adoptE = placeholders.adopt_evs(Ldata, meterdf, E_g)
        adoptS = placeholders.adopt_solar(Ldata, solardf, S_g)
        print(f"adoption computed", time.perf_counter() - t0)
        Hsize += placeholders.size_heatpumps(Ldata, adoptH).values
        Esize += placeholders.size_evs(Ldata, adoptE).values
        Ssize += placeholders.size_solar(Ldata, adoptS).values
        print(f"sizes updated", time.perf_counter() - t0)

        weather = weather_data.load_data.generate()


        LH = placeholders.generate_heatpump_load_profile(weather)[:, np.newaxis] # just one profile
        LE = placeholders.generate_ev_load_profile(Esize)
        LS = sun_model.generate()[:, np.newaxis] # just one profile
        L0 = placeholders.generate_background_profile(Ldata)
        L = Hsize*LH + LE + Ssize*LS + L0
        print(f"profiles generated", time.perf_counter() - t0)

        # Transformer loads
        L_tr = loads_to_transformers.meters_to_transformers_tuples(load_map, L).values
        #L_tr = L.values @ load_map_sparse
        print(f"transformer loads calculated", time.perf_counter() - t0)

        # Transformer aging
        hotspot = transformer_aging.temperature_equations(L_tr, weather, T0=T0)
        print(f"hotspot T calculated", time.perf_counter() - t0)
        T0 = hotspot[-1] # for iterating multiple years if desired
        aging = transformer_aging.effective_aging(hotspot)
        aging += age_offset
        age_offset = aging[-1]
        age_curves.append(aging)
        print(f"year {year} done", time.perf_counter() - t0)
        
    full_age_curve = np.concatenate(age_curves, axis=0)
    
if __name__ == "__main__":
    
    run_instance(nyears=2, year0=2025)
