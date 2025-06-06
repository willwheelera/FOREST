import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append("../")

substation_name = "Alburgh"
pkl_file = f"data/{substation_name}/{substation_name}_Model.pkl"
with open(pkl_file, "rb") as file:
    ps_model = pickle.load(file)

vals = [(b.name, b.ratedKVA) for i, b in enumerate(ps_model.Branches) if b.type=="transformer"]
tf_name, ratedKVA = list(zip(*vals))
df = pd.DataFrame({"tf_name": tf_name, "ratedKVA": ratedKVA})
df.to_csv("transformer_ratings.csv")
