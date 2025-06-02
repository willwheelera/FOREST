import pandas as pd
import pickle
import time
import os
import sys
sys.path.append("../")
import GLM_Tools
import PowerSystemModel as psm
import visualize_network

def generate_transformer_map(substation_name="South_Alburgh"):
    os.chdir("../")
    pkl_file = f"Feeder_Data/{substation_name}/Python_Model/{substation_name}_Model.pkl"
    load_dict = pd.read_csv(f"Feeder_Data/{substation_name}/meter_number_data.csv")
    gen_dict = pd.read_csv(f"Feeder_Data/{substation_name}/gen_meter_number_data.csv")

    with open(pkl_file, "rb") as file:
        ps_model = pickle.load(file)

    load_map = map_meters_to_transformers(ps_model, load_dict, meter_type="load")
    gen_map = map_meters_to_transformers(ps_model, gen_dict, meter_type="gen")
    transformer_map = dict(load_map=load_map, gen_map=gen_map)
    # Save updated pkl file
    with open(f"Feeder_Data/{substation_name}/Python_Model/transformer_map.pkl", "wb") as file:
        pickle.dump(transformer_map, file)

def map_meters_to_transformers(ps_model, meter_dict, meter_type="load"):
    id_key_dict = {"load": "Service Number", "gen": "Object ID"}
    id_key = id_key_dict[meter_type.lower()]
    # Loop through AMI loads
    pairs = []
    for index, row in meter_dict.iterrows():
        service_num = row[id_key]
        load_name = f"_{service_num}_cons"
        meter_num = row["Meter Number"]
        pairs.extend(find_transformers(ps_model, load_name, meter_num, meter_type))
    return pairs


def find_transformers(ps_model, load_name, meter_num, meter_type="load"):
    pairs = []
    load_dict = ps_model.Load_Dict if meter_type == "load" else ps_model.Generator_Dict
    if load_name not in load_dict.keys():
        return []
    load = load_dict[load_name]
    node = ps_model.Node_Dict[load.parent]
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
            pairs.append((meter_num, branch.name))
        from_node = branch.from_node
        if from_node not in ps_model.Node_Dict:
            from_node = psm.find_node_parent(from_node, ps_model.Node_Dict, ps_model.Shunt_Dict)
        node = ps_model.Node_Dict[from_node]
    return pairs
    
if __name__ == "__main__":
    t0 = time.perf_counter()
    generate_transformer_map()
    print("time", time.perf_counter() - t0)
