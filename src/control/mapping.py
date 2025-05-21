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