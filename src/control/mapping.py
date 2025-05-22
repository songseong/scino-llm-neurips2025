from src.data import get_dataset

def mapping_fun(dataset_name):
    
    _, _, node_names = get_dataset(dataset_name)

    index_to_name = {i: name for i, name in enumerate(node_names)} 
    name_to_index = {name: i for i, name in enumerate(node_names)} 

    def normalize_name(name):
        return name.lower().replace("/", "_")  # '/' → '_'

    name_to_index = {normalize_name(name): i for i, name in enumerate(node_names)}

    def to_index(name):
        return name_to_index.get(normalize_name(name), None)

    def to_name(index):
        return index_to_name.get(index, None)  
    
    return to_index, to_name

def get_mapped_index(var, dataset_name, to_index):
    if dataset_name == "physics":
        return to_index(physics_name_map[var])
    else:
        return to_index(var.lstrip("_"))

def get_mapped_name(index, dataset_name, to_name):
    raw_name = to_name(index)
    if dataset_name == "physics":
        reverse_map = {v: k for k, v in physics_name_map.items()}
        return reverse_map.get(raw_name, f"_UNKNOWN_{index}")
    else:
        return f"_{raw_name}"

physics_name_map = {
    "_Total_Solar_Irradiance": "Total solar irradiance",
    "_Wind_Speed": "Wind Speed",
    "_Surface_Air_Temperature": "Surface Air Temperature",
    "_Rate_of_Evaporation": "Rate of Evaporation",
    "_Rainfall": "Rainfall",
    "_Moisture_Content_of_object": "Moisture Content of object",
    "_Weight_of_object": "Weight of object"
}

if __name__ == "__main__":
    to_index, to_name = mapping_fun('physics', num_samples=1)
    
    print("=== Mapping: Index → Node Name ===")
    for index in range(7):  
        name = to_name(index)
        print(f"Index {index} -> Node name: '{name}'")

    print("\n=== Mapping: Node Name → Index ===")
    for index in range(7):
        name = to_name(index)
        node_index = to_index(name)
        print(f"Node name '{name}' -> Index: {node_index}")