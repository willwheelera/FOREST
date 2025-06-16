import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import weather_data.load_data
import os
import time

def read_transformer_ratings(feeder="Alburgh"):
    path = f"data/{feeder}/"
    tf_fname = "normalized_transformer_loading.parquet"
    
    # Read in transformer rated kVA
    ratings = pd.read_csv(path+"transformer_ratings.csv", index_col=0)
    newratings = pd.read_csv(path+"2025_transformer_ratings.csv", index_col=0)
    ratings = ratings.set_index("tf_name")
    newratings = newratings.set_index("gs_equipment_location")

    # Try to fix incorrect values
    bigkeys = ['E72805100040038', 'E72805103079841', 'E72805103080326',
   'E72805103080328', 'E72805103097973', 'E72805203078657',
   'E72805203078706', 'E72805203078859', 'E72805203078866',
   'E72805203079531', 'E72805203079711', 'E72805203080317']
    ratings.loc[bigkeys, "ratedKVA"] = newratings.loc[bigkeys, "gs_rated_kva"]
    ratings.loc["E72805203096474", "ratedKVA"] = 100
    return ratings
