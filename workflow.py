import pandas as pd
import transformer_aging
import time
import numpy as np
import weather_data.load_data
import placeholders

# General outline

def run_instance(nyears=20, year0=2025):
    """
    Ldata: past (2024) one-year dataframe, (hours, meters)
    meterdf: dataframe with meter numbers and device adoption status
    solardf: dataframe with meter numbers and solar status
    mapfile: information about which meters connect to which transformers
    """
    # Read in load data -- may want to use raw data instead
    fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
    mapfile = "" # map meters to transformers
    Ldata = pd.read_parquet(fname)
    meterdf, keys = device_data.get_meter_data()
    solardf = meterdf # TODO.get_solar_data()
    
    for year in np.arange(nyears) + year0:
        H_g = placeholders.growth_rate_heatpumps(year)
        E_g = placeholders.growth_rate_evs(year)
        S_g = placeholders.growth_rate_solar(year)

        adoptH = placeholders.adopt_heatpumps(Ldata, meterdf, H_g)
        adoptE = placeholders.adopt_evs(Ldata, meterdf, E_g)
        adoptS = placeholders.adopt_solar(Ldata, solardf, S_g)
        H = placeholders.size_heatpumps(Ldata, adoptH)
        E = placeholders.size_evs(Ldata, adoptE)
        S = placeholders.size_solar(Ldata, adoptS)

        weather = weather_data.load_data.generate()

        LH = TODO.generate_heatpump_load_profile()
        LE = TODO.generate_ev_load_profile()
        LS = TODO.generate_solar_gen_profile()
        L0 = TODO.generate_background_profile(Ldata)
        L = H*LH + E*LE + S*LS + L0

        # Transformer loads
        L_tr = TODO.loads_to_transformers(L, meterdf, mapfile)

        # Transformer aging
        hotspot = transformer_aging.temperature_equations(L_tr, weather, T0=T0)
        T0 = hotspot[-1] # for iterating multiple years if desired
        aging = transformer_aging.effective_aging(hotspot)


