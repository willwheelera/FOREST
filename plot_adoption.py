import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from device_data import clean_data


def run():
    datapath = "/home/wwheele1/FOREST/data/"
    device_df = pd.read_excel(datapath+"All T3 through 2024.xlsx")
    clean_data(device_df)

    device_df = device_df[device_df["Substation"] == 28]

    #key = "home charger"
    key = "cchp"
    chargers = device_df[device_df["Measure"] == key]

    isstr = chargers["Year"].str.isnumeric() == False
    index = chargers.index[isstr]
    chargers.loc[index, "Year"] = [int(s[-4:]) for s in chargers["Year"][isstr]]
    chargers["Year"] = chargers["Year"].astype(int)
    years = chargers["Year"].values
    vals, counts = np.unique(years, return_counts=True)
    print(vals)
    print(counts)

    dates = pd.to_datetime(chargers["Date of purchase"])
    dates = dates.sort_values()
    plt.plot(dates, np.arange(len(dates)))
    plt.title(key)
    plt.show()


if __name__ == "__main__":
    run()
