import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A, B = 3.8421506567965565e-06, -78.80629154833822
C, D = 2.7476981127290047e-06, 42.493498449250744

def fit_constants():
    df = pd.read_csv("data/Alburgh/2025_transformer_ratings.csv")
    df = df.set_index("gs_equipment_location")

    test_locs = []
    test_locs.append(dict(loc="E72805203079240", lat=45.012434, long=-73.210921))
    test_locs.append(dict(loc="E72805203078806", lat=44.982615, long=-73.341124))
    test_locs.append(dict(loc="E72805103080060", lat=44.835578, long=-73.359238))
    test_locs.append(dict(loc="E72805003081423", lat=44.824280, long=-73.275468))
    test_locs.append(dict(loc="E72805200040510", lat=44.912476, long=-73.268812))
    tl = pd.DataFrame(test_locs)
    df = df.loc[tl["loc"]]

    a, b = np.polyfit(df["x"], tl["long"], 1)
    c, d = np.polyfit(df["y"], tl["lat"], 1)
    print(a, b)
    print(c, d)

def print_locs():
    df = pd.read_csv("data/Alburgh/2025_transformer_ratings.csv")
    df = df.set_index("gs_equipment_location")
    df.drop(columns="Unnamed: 0", inplace=True)
    locs = ["E72805103079841", "E72805103080328"]
    df = df.loc[locs]
    df["lat"] = C * df["y"] + D
    df["long"] = A * df["x"] + B
    print(df)

if __name__ == "__main__":
    print_locs()
