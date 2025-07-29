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


# General outline

def run_2024():
    """
    Ldata: past (2024) one-year dataframe, (hours, meters)
    meterdf: dataframe with meter numbers and device adoption status
    solardf: dataframe with meter numbers and solar status
    mapfile: information about which meters connect to which transformers
    """
    timer = Timer(30)
    timer.print("Starting")
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

    age_curves = []
    load_curves = []
    if mapfile.endswith("json"):
        with open(mapfile, "r") as f:
            load_map = json.load(f)
        age_offset = np.zeros(len(load_map.keys()))
    else:
        with open(mapfile, "rb") as f:
            load_map = pickle.load(f)["load_map"]
        #load_map = [l for l in load_map if l[0] in Ldata.columns]
        m2t_map = loads_to_transformers.meter_to_transformer_matrix(load_map, Ldata.columns)
        age_offset = np.zeros(m2t_map.shape[1])
    tf_ratings = read_in_data.read_transformer_ratings("Alburgh")
    tf_ratings = tf_ratings.loc[m2t_map.columns]
    TF_RATINGS = tf_ratings["ratedKVA"].values

    timer.print("data loaded")#, time.perf_counter() - t0)
    T0 = 110

    timer.reset()#t0 = time.perf_counter()
    timer.print(f"Reset timer")

    weather = weather_data.load_data.generate()
    pfactor = 1 - 0.1 * (~meterdf["solar"]).astype(float) # assume inverter PF is 1
    L = Ldata / pfactor # power factor

    # Transformer loads
    #L_tr = loads_to_transformers.meters_to_transformers_tuples(load_map, L).values
    #L_tr = m2t_mapper.meters_to_transformers(L)
    L_tr = L.values @ m2t_map.values
    L_tr = (L_tr / TF_RATINGS)
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
    timer.print(f"done")#, time.perf_counter() - t0)
        
    full_age_curve = np.concatenate(age_curves, axis=0)
    df = pd.DataFrame(data=full_age_curve, columns=m2t_map.columns)
    df.to_parquet(f"output/alburgh_tf_aging_2024.parquet")
    full_load_curve = np.concatenate(load_curves, axis=0)
    df = pd.DataFrame(data=full_load_curve, columns=m2t_map.columns)
    df.to_parquet(f"output/alburgh_tf_load_2024.parquet")
    # TODO save to hdf instead of multiple parquet files

    sizes = meterdf[["cchp", "home charger", "solar"]]
    tf_device_sizes = m2t_map.T @ sizes
    tf_device_sizes["age"] = df.iloc[-1]
    tf_device_sizes.to_parquet(f"output/alburgh_tf_devices_2024.parquet")
    
if __name__ == "__main__":
    
    run_2024()
