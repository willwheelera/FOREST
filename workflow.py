import pandas as pd
import transformer_aging
import time
import numpy as np
import weather_data.load_data
import placeholders
import sun_model


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
    Hsize, Esize, Ssize = np.zeros(Ldata.shape[1]), np.zeros(Ldata.shape[1]), np.zeros(Ldata.shape[1])
    
    for year in np.arange(nyears) + year0:
        H_g = placeholders.growth_rate_heatpumps(year)
        E_g = placeholders.growth_rate_evs(year)
        S_g = placeholders.growth_rate_solar(year)

        adoptH = placeholders.adopt_heatpumps(Ldata, meterdf, H_g)
        adoptE = placeholders.adopt_evs(Ldata, meterdf, E_g)
        adoptS = placeholders.adopt_solar(Ldata, solardf, S_g)
        Hsize += placeholders.size_heatpumps(Ldata, adoptH)
        Esize += placeholders.size_evs(Ldata, adoptE)
        Ssize += placeholders.size_solar(Ldata, adoptS)

        weather = weather_data.load_data.generate()

        LH = TODO.generate_heatpump_load_profile()
        LE = TODO.generate_ev_load_profile()
        LS = sun_model.generate()[:, np.newaxis] # just one profile
        L0 = placeholders.generate_background_profile(Ldata)
        L = Hsize*LH + Esize*LE + Ssize*LS + L0

        # Transformer loads
        L_tr = TODO.loads_to_transformers(L, meterdf, mapfile)

        # Transformer aging
        hotspot = transformer_aging.temperature_equations(L_tr, weather, T0=T0)
        T0 = hotspot[-1] # for iterating multiple years if desired
        aging = transformer_aging.effective_aging(hotspot)


