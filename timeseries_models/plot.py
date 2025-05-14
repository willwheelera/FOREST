import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def avg_day(x, **kwargs):
    # x is (8760, ncurves)
    x = x.reshape(365, 24, -1).mean(axis=0)
    t = np.arange(24)
    plt.plot(t, x, **kwargs)
    
def avg_week(x, **kwargs):
    # x is (8760, ncurves)
    x = x[:364*24].reshape(52, 7, 24, -1).mean(axis=(0, 2))
    t = np.arange(7)
    plt.plot(t, x, **kwargs)
    
def avg_over_day(x, **kwargs):
    # x is (8760, ncurves)
    x = x.reshape(365, 24, -1).mean(axis=1)
    t = np.arange(365)
    plt.plot(t, x, **kwargs)
    
