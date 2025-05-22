import numpy as np

def make_leaf_node_prompt(variables, data_description=None):
    file_path = "src/control/prompt.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()
    active_nodes = [f"node{var}" for var in variables.keys()]
    variable_description_str = "\n".join([
        f'"node{var}": "{desc}"' for var, desc in variables.items()
    ])
    prompt = template.format(
        active_nodes=active_nodes,
        data_description=data_description,
        variable_description_str=variable_description_str
    )
    return prompt

def softmax(x, diffan_list, tau=1.0):
    tau_x = np.array([val ** tau for val in x])
    sum_x_tau = np.sum(np.array([i ** tau for i in diffan_list]))  
    softmax_result = tau_x / sum_x_tau  
    return softmax_result

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()
            
def convert_to_builtin(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

## partial context setting
def mapping_mask(dataset_name, config_file):
    
    node_names = config_file[dataset_name]["variables"].keys()

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

def restore_variable_name(dataset,config,config_base,masked_name):
    
    to_index_masked, _ = mapping_mask(dataset, config)
    _, to_name1 = mapping_mask(dataset, config_base)

    index = to_index_masked(masked_name)
    name = to_name1(index)

    return name

def change_variable_name(dataset,config,config_base,name):

    _, to_name_masked = mapping_mask(dataset, config)
    to_index1, _ = mapping_mask(dataset, config_base)

    index = to_index1(name)
    masked_name = to_name_masked(index)

    return masked_name