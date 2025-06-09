import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

names = ["BURLINGTON INTERNATIONAL AIRPORT, VT US"]#, "ESSEX JUNCTION VERMONT, VT US", "SOUTH HERO, VT US"]

def load_data(path=""):
    fname = path+"weather_data/burlington_weather_2024.csv"
    df = pd.read_csv(fname)

    df = df[df["NAME"].isin(names)].drop(columns="NAME")
    df.reset_index(inplace=True)
    df["DATE"] = pd.to_datetime(df["DATE"])

    cols = ["DATE", "TAVG", "TMIN", "TMAX"]
    df = df[cols]
    return df


def interpolate(df):
    t = np.arange(8760)
    base = (1 - np.cos(2*np.pi*t/24)) / 2
    min_ = np.zeros_like(base)
    min_[5::24] = df["TMIN"].values[:365]
    min_s = np.convolve(min_, base[:24])[12:8772]
    max_ = np.zeros_like(base)
    max_[17::24] = df["TMAX"].values[:365]
    max_s = np.convolve(max_, base[:24])[12:8772]
    avg_shift = smooth_avg_shift(df)[7:8767]
    return min_s + max_s + avg_shift
    #plt.plot(np.arange(5, 8760, 24), df["TMIN"][:365])
    #plt.plot(np.arange(17, 8760, 24), df["TMAX"][:365])
    #plt.plot(min_s + max_s)
    #plt.plot(min_s + max_s + avg_shift)
    #plt.plot(avg_shift)
    #plt.show()
    
def smooth_avg_shift(df):
    diff = df["TMAX"] - df["TMIN"]
    avg_diff = df["TAVG"] - (df["TMIN"] + df["TMAX"]) / 2
    coeffs = smooth_avg_shift_coeffs()
    avg_diff = np.clip(avg_diff / coeffs.sum(), -diff/2, diff/2)
    correction = np.zeros(24)
    t = np.linspace(0, 2*np.pi, 24, endpoint=False)
    for i, c in enumerate(coeffs):
        correction += np.sin((1+i)*t)**2 * c 
    shift = avg_diff.values[:, np.newaxis] * correction
    return shift.ravel()

def smooth_avg_shift_coeffs():
    # x is average. 
    # return 24 hour curve that smoothly, symmetrically shifts a cosine to the new average while preserving max and min
    # idea: move towards zero curvature at max (don't exceed)
    #       cancel other higher order terms as well
    # cos x    =              1 - 1/2 x^2 +    1/4! x^4 -      1/6! x^6 +        1/8! x^8 - 
    # sin^2  x = 1/2 - 1/2 cos(2x) =  x^2 -    8/4! x^4 +     32/6! x^6 -      128/8! x^8 + ...
    # sin^2 2x = 1/2 - 1/2 cos(4x) = 4x^2 - 8*16/4! x^4 +  32*64/6! x^6 -  128*256/8! x^8 + ...
    # sin^2 3x = 1/2 - 1/2 cos(6x) = 9x^2 - 8*81/4! x^4 + 32*729/6! x^6 - 128*6561/8! x^8 + ...
    #   avg is 1/2
    #   cos(x) - 1/2 cos^2 x = 1/2 + 3/8 x^4 + ... max increase of 1/2
    # a sin^2(x) + b sin^2(2x) + c sin^2(3x)
    # a + 4b + 9c <= 1/2
    # a + 16b + 81c >= 1/8
    # a + 64b + 729c <= 1/32
    n = 6
    base = np.arange(1, n+1)
    A = base ** (2*base[:, np.newaxis])
    b0 = 2. ** (-base*2 + 1)
    b = b0 * (1 + ((-1)**base +.5) / 10) 
    coeffs = np.linalg.lstsq(A, b)[0]
    return coeffs

def plot_smooth_avg_shift(shift=0.5):
    coeffs = smooth_avg_shift_coeffs()
    t = np.linspace(0, 2*np.pi, 25)
    base = np.cos(t)
    
    coeffs = coeffs * shift / coeffs.sum()
    correction = np.zeros_like(base)
    for i, c in enumerate(coeffs):
        correction += np.sin((1+i)*t)**2 * c 
    plt.plot(base )
    plt.plot(base + correction)
    plt.plot(correction)
    plt.axhline(y=coeffs.sum())
    plt.show()

if __name__ == "__main__":
    df = load_data("../")
    interpolate(df)
