import numpy as np
import matplotlib.pyplot as plt

# Weibull model for daily miles (km) driven, http://dx.doi.org/10.1016/j.trb.2017.04.008
# f(r) = (beta/tau) (r/tau)^{beta-1} e^{-(r/tau)^beta}


def generate_ev_load_profile(Esize, seed=1, charging_rate=7.2, ndays=365):
    Esize = np.asarray(Esize)
    hasEV = Esize > 0
    n = hasEV.sum()
    rng = np.random.default_rng(seed)
    daily_load = generate_daily_charging_load(Esize, ndays, seed)
    load_profile = np.zeros((len(Esize), ndays, 24))
    for ev, load in zip(np.where(hasEV)[0], daily_load.T):
        days = np.where(load > 0)[0]
        n_ = len(days)
        start_time = np.floor(rng.normal(6.5, 1.5, size=n_)).astype(np.int64)
        duration = np.ceil(load[days] / charging_rate).astype(np.int64)
        for d, s, t in zip(days, start_time, duration):
            load_profile[ev, d, s:s+t] = charging_rate
    return np.roll(load_profile.reshape(-1, ndays*24), 12, axis=1)

# Want to calculate charge depleted
# Parameters: 
#   car efficiency miles/kWh (2-5)
#   charge threshold (40-60) # when to recharge
#   tau (30-55) # avg miles driven
#   beta 1.53 #(1.4-1.7)
#   battery size
#   charging efficiency 85%
def generate_daily_charging_load(battery_size, ndays, seed=1):
    battery_size = battery_size[battery_size > 0]
    n = len(battery_size)
    rng = np.random.default_rng(seed)
    tau = rng.uniform(30, 55, size=n) # tau and mpkWh don't really need to be separated
    mpkWh = rng.uniform(2, 5, size=n)
    scale = tau / mpkWh / battery_size
    beta = 1.53 # 
    charge_threshold = rng.uniform(0.4, 0.6, size=n) # how much battery drain before charging again
    eff = 0.85
    
    charge_used = rng.weibull(beta, size=(ndays, n)) * scale
    load = np.zeros((ndays, n))
    cum = np.zeros(n) + 0.3 # start off with partial battery
    for t, c in enumerate(charge_used):
        cum += c
        cum = np.minimum(cum, 1.)
        recharge = cum > charge_threshold
        load[t, recharge] = cum[recharge] * battery_size[recharge] / eff
        cum[recharge] = 0.
    
    return load


def plot_distribution():
    tau = 1#42.6
    beta = 1.53
    for beta in np.linspace(1.4, 1.7, 5):
        f = lambda r: (beta/tau) *(r/tau)**(beta-1) * np.exp(-(r/tau)**beta)
        d = np.linspace(0, 2.5*tau, 100)
        plt.plot(d, f(d), label=beta)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Esize = np.array([50, 60, 70, 80, 90])
    n = len(Esize)
    ndays = 7
    load = generate_ev_load_profile(Esize, seed=1, charging_rate=7.2, ndays=ndays)
    daily_load = generate_daily_charging_load(Esize, ndays, seed=1)

    rng = np.random.default_rng(1)
    tau = rng.uniform(30, 55, size=n) # tau and mpkWh don't really need to be separated
    mpkWh = rng.uniform(2, 5, size=n)
    scale = tau / mpkWh / np.asarray(Esize)
    charge_threshold = rng.uniform(0.4, 0.6, size=n) # how much battery drain before charging again
    charge_used = np.cumsum(rng.weibull(1.53, size=(ndays, n)) * scale, axis=0) + 0.3

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
