import pandas as pd
import pickle
import time
import os
import sys
sys.path.append("../MAPLE_BST_Demo/")
import GLM_Tools
import GLM_Tools.PowerSystemModel as psm
import visualize_network

def generate_transformer_map(substation_name="Alburgh"):
    #os.chdir("../")
    pkl_file = f"data/{substation_name}/{substation_name}_Model.pkl"
    load_dict = pd.read_parquet(f"data/VEC_meter_number_data.parquet")
    gen_dict = pd.read_parquet(f"data/VEC_gen_meter_number_data.parquet")

    with open(pkl_file, "rb") as file:
        ps_model = pickle.load(file)
    load_map = map_meters_to_transformers(ps_model, load_dict, meter_type="load")
    gen_map = map_meters_to_transformers(ps_model, gen_dict, meter_type="gen")
    transformer_map = dict(load_map=load_map, gen_map=gen_map)
    # Save updated pkl file
    with open(f"data/{substation_name}/transformer_map.pkl", "wb") as file:
        pickle.dump(transformer_map, file)

def map_meters_to_transformers(ps_model, meter_dict, meter_type="load"):
    # Loop through AMI loads
    meter_dict = meter_dict[meter_dict["Substation"] == '28']
    pairs = []
    for index, row in meter_dict.iterrows():
        pairs.extend(find_transformers(ps_model, row, meter_type))
    return pairs


id_key_dict = {"load": "Service Number", "gen": "Object ID"}
def find_transformers(ps_model, row, meter_type="load"):
    id_key = id_key_dict[meter_type.lower()]
    if meter_type=="load":
        load_name = f"_{row['Service Number']}_cons"
        load_dict = ps_model.Load_Dict
    elif meter_type=="gen":
        load_name = f"gene_{row['Object ID']}_negLdGen"
        load_dict = ps_model.Generator_Dict
    if load_name not in load_dict.keys():
        return []
    load = load_dict[load_name]
    node = ps_model.Node_Dict[load.parent]
    pairs = []
    while len(node.incoming_branches) > 0:
        if len(node.incoming_branches) > 1:
            print(f"{node.name} has {len(node.incoming_branches)} incoming branches")
            print(node.X_coord, node.Y_coord)
            print("incoming")
            for bind in node.incoming_branches:
                b = ps_model.Branches[bind]
                print(bind, b.name, b.type, b.from_node)
        branch = ps_model.Branches[node.incoming_branches[0]]
        if branch.type == "transformer":
            pairs.append((int(row["Meter Number"]), branch.name))
        from_node = branch.from_node
        if from_node not in ps_model.Node_Dict:
            from_node = psm.find_node_parent(from_node, ps_model.Node_Dict, ps_model.Shunt_Dict)
        node = ps_model.Node_Dict[from_node]
    return pairs
    
if __name__ == "__main__":
    t0 = time.perf_counter()
    generate_transformer_map()
    print("time", time.perf_counter() - t0)
