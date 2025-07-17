import pandas as pd
import transformer_aging
import time

# General outline

# Read in load data -- may want to use raw data instead
fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
mapfile = "" # map meters to transformers
Ldata = pd.read_parquet(fname)
meterdf, keys = device_data.get_meter_data()
solardf = TODO.get_solar_data()

# Generate adoption
H = TODO.predict_heatpumps(Ldata, meterdf)
E = TODO.predict_evs(Ldata, meterdf)
S = TODO.predict_solar(Ldata, solardf)

# Read in weather data
weather = weather_data.load_data.load_data()
weather = weather_data.load_data.interpolate(weather)
weather = (weather - 32) * 5/9

# Generate profiles
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


