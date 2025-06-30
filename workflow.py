

# General outline

# Read in load data -- may want to use raw data instead
fname = "data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
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
LH = TODO.
