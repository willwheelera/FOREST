import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append("../")

substation_name = "Alburgh"
pkl_file = f"data/{substation_name}/{substation_name}_Model.pkl"
with open(pkl_file, "rb") as file:
    ps_model = pickle.load(file)

vals = [(b.name, b.ratedKVA, b.X_coord, b.Y_coord) for i, b in enumerate(ps_model.Branches) if b.type=="transformer"]
tf_name, ratedKVA, x, y = list(zip(*vals))
df = pd.DataFrame({"tf_name": tf_name, "ratedKVA": ratedKVA, "x": x, "y": y})
df.to_csv(f"data/{substation_name}/transformer_ratings.csv")

# need to check gs_tran_kva_a, etc - what happens if two or three of (a, b, c) are present? how many gs_rated_kva don't match?
