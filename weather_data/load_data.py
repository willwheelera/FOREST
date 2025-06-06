import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

names = ["BURLINGTON INTERNATIONAL AIRPORT, VT US"]#, "ESSEX JUNCTION VERMONT, VT US", "SOUTH HERO, VT US"]

def load_data():
    fname = "/home/wwheele1/FOREST/weather_data/burlington_weather_2024.csv"
    df = pd.read_csv(fname)

    df = df[df["NAME"].isin(names)].drop(columns="NAME")
    df.reset_index(inplace=True)
    df["DATE"] = pd.to_datetime(df["DATE"])

    cols = ["DATE", "TAVG", "TMIN", "TMAX"]
    df = df[cols]
    return df


