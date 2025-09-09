import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Weibull model for daily miles (km) driven, http://dx.doi.org/10.1016/j.trb.2017.04.008
# Ploetz et al. 2017 
# f(r) = (beta/tau) (r/tau)^{beta-1} e^{-(r/tau)^beta}

# Parameters
# Random parameters drawn uniformly between lims
BATTERY_LIMS = (50., 120.) # EV size kWh
TAU_LIMS = (40., 60.) # ~avg daily km (~10% over because of beta)
MPKWH_LIMS = (2., 5.) # vehicle efficiency miles/kWh
RECHARGE_LIMS = (0.4, 0.6) # when to recharge, % battery used

# Fixed parameters
BETA = 1.53 # Weibull shape parameter for miles driven
EFF = 0.85 # charging efficiency
INIT_BATTERY = 0.3 # initial battery used (1 - SOC) at day 0
CHARGING_RATE = 7.2 # kW, level 2 charging rate

# Gaussian parameters
PLUGIN = (6.5, 1.5) # mean (PM), std (h) of plug-in time

def generate_parameters(inds, seed=1):
    n = len(inds)
    rng = np.random.default_rng(seed)
    battery = rng.uniform(*BATTERY_LIMS, size=n) 
    tau = rng.uniform(*TAU_LIMS, size=n) / 1.609 #  km -> miles 
    mpkWh = rng.uniform(*MPKWH_LIMS, size=n) 
    ch_at = rng.uniform(*RECHARGE_LIMS, size=n) 
    scale = tau / mpkWh / battery
    parameters = np.stack([inds, battery, scale, ch_at], axis=0)
    return parameters

def generate_ev_load_profile(Eparams, n, seed=1, ndays=365):
    daily_load = generate_daily_charging_load(Eparams, ndays, seed)
    inds = Eparams[0].astype(np.int64)
    load_profile = np.zeros((n, ndays* 24)) # 24 h/day, fixed
    rng = np.random.default_rng(seed)
    tmp = np.zeros((ndays, 24))
    for ev, load in zip(inds, daily_load.T):
        days = np.where(load > 0)[0]
        n_ = len(days)
        start_time = np.floor(rng.normal(*PLUGIN, size=n_)).astype(np.int64)
        duration = np.ceil(load[days] / CHARGING_RATE).astype(np.int64)
        tmp[:] = 0.
        for d, s, t in zip(days, start_time, duration):
            tmp[d, s:s+t] = CHARGING_RATE
        load_profile[ev] = np.roll(tmp.reshape(-1), 12)
    return load_profile.T

def generate_daily_charging_load(params, ndays, seed=1):
    _, battery_size, scale, charge_threshold = params
    n = params.shape[1]
    rng = np.random.default_rng(seed)
    
    charge_used = rng.weibull(BETA, size=(ndays, n)) * scale
    load = np.zeros((ndays, n))
    cum = np.zeros(n) + INIT_BATTERY # start off with partial battery
    for t, c in enumerate(charge_used):
        cum += c
        cum = np.minimum(cum, 1.)
        recharge = cum > charge_threshold
        load[t, recharge] = cum[recharge] * battery_size[recharge] / EFF
        cum[recharge] = 0.
    
    return load


def plot_distribution():
    tau = 1#42.6
    for beta in np.linspace(1.4, 1.7, 5):
        f = lambda r: (beta/tau) *(r/tau)**(beta-1) * np.exp(-(r/tau)**beta)
        d = np.linspace(0, 2.5*tau, 100)
        plt.plot(d, f(d), label=beta)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    n = 5
    inds = np.arange(n)
    params = generate_parameters(inds, seed=1)
    ndays = 7
    load = generate_ev_load_profile(params, n, seed=1, ndays=ndays)
    daily_load = generate_daily_charging_load(params, ndays, seed=1)

    scale = params[2]
    charge_threshold = params[3]
    rng = np.random.default_rng(1)
    charge_used = np.cumsum(rng.weibull(BETA, size=(ndays, n)) * scale, axis=0) + INIT_BATTERY

    fig, axs = plt.subplots(3, 1)
    t = np.arange(ndays*24) / 24
    axs[0].plot(charge_used)
    axs[0].axhline(y=.5, c="k", lw=0.5)
    axs[0].set_xlim([0, 7])
    axs[1].plot(daily_load)
    axs[1].set_xlim([0, 7])
    axs[2].plot(t, load.T)
    axs[2].set_xlim([0, 7])
    plt.show()
