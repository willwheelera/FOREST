import numpy as np


def estimate_logistic_rate(year, f_now, y50, ynow=2024):
    x = -np.log(1/f_now - 1)
    a = (ynow - y50) / x
    return 1 / (4*a*np.cosh((year-y50)/(2*a))**2)

def growth_rate_heatpumps(year):
    # Alburgh
    f_now = 0.11 # estimated penetration from current data
    y50 = 2050 # estimate year of 50% hp penetration
    return estimate_logistic_rate(year, f_now, y50)

def growth_rate_evs(year):
    # Alburgh
    f_now = 0.005 # estimated penetration from current data
    y50 = 2060 # estimate year of 50% ev penetration
    return estimate_logistic_rate(year, f_now, y50)

def growth_rate_solar(year):
    # Alburgh
    f_now = 0.02 # estimated penetration from current data
    y50 = 2060 # estimate year of 50% solar penetration
    return estimate_logistic_rate(year, f_now, y50)

def _adopt(can_adopt, key, meterdf, growth):
    can_adopt = can_adopt & ~meterdf[key] # doesn't already have
    add_random = np.random.random(can_adopt.sum()) < growth
    can_adopt[can_adopt] = add_random
    meterdf.loc[:, key] = meterdf[key] | can_adopt
    return can_adopt

def adopt_heatpumps(Ldata, meterdf, H_g):
    can_adopt = Ldata[:60*24].abs().mean(axis=0) > 0.2 # not abandoned in winter
    return _adopt(can_adopt, "cchp", meterdf, H_g)

def adopt_evs(Ldata, meterdf, E_g):
    can_adopt = Ldata[150*24:260*24].abs().mean(axis=0) > 0.2 # used in summer
    return _adopt(can_adopt, "home charger", meterdf, E_g)

def adopt_solar(Ldata, solardf, S_g):
    can_adopt = Ldata[150*24:260*24].abs().mean(axis=0) > 0.2 # used in summer
    return _adopt(can_adopt, "solar", solardf, S_g)

def size_heatpumps(Ldata, adopt):
    newcols = Ldata.columns[adopt]
    new_size = adopt.astype(float)
    new_size[adopt] = Ldata[newcols].sum(axis=0) / 365 / 4 # arbitrary estimate of hp size in kW(electricity)
    return new_size

def size_evs(Ldata, adopt):
    new_size = adopt.astype(float)
    new_size[adopt] = np.random.uniform(50, 100, size=adopt.sum()) # car battery size
    # 7.2 # charger peak size in kW
    return new_size

def size_solar(Ldata, adopt):
    newcols = Ldata.columns[adopt]
    new_size = adopt.astype(float)
    tmp = Ldata[newcols].sum(axis=0) / 365 / 3 # arbitrary estimate of solar size in kW_peak
    new_size[adopt] = np.clip(tmp, a_max=25., a_min=0.)
    return new_size

def generate_background_profile(Ldata):
    return Ldata.copy()

def generate_heatpump_load_profile(temp):
    # assume linear efficiency centered at 22C, up to 40C difference
    # assume duty cycle averages out within the hour
    # assume no size-dependence - just one profile
    Tset = 22
    power_frac = np.abs(temp - Tset) / 40
    return np.clip(power_frac, 0, 1)

def generate_ev_load_profile(Esize):
    # suppose charging 80% battery every night
    charge_rate = 7.2 # kW, level 2 charger
    hasEV = Esize > 0
    n = hasEV.sum()
    start_time = np.random.randint(6, 12, size=n) # pm
    duration = np.ceil(Esize * 0.8 / charge_rate).astype(int)
    profiles = np.zeros((len(Esize), 365, 24)) # start at 12pm, shift later
    for i, j in enumerate(np.where(hasEV)[0]):
        t0 = start_time[i]
        profiles[j, :, t0:t0+duration[j]] = charge_rate
    profiles = profiles.reshape(-1, 8760)
    profiles = np.roll(profiles, 12, axis=1).T
    return profiles

