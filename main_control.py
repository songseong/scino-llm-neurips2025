from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import torch.nn.functional as F
from src.control.mapping import mapping_fun
from src.utils import num_errors, dict2namespace
from src.data import get_dataset
from datetime import datetime
import json
from src.control.ordering import ensemble_topological_ordering
import sys
import yaml
import os
import hashlib
import pickle
from dotenv import load_dotenv

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


def make_leaf_node_prompt1(variables: dict) -> str:

    formatted_vars = [
        f"node{short_name} ({desc})" for short_name, desc in variables.items()
    ]

    prompt = f""" In a Bayesian network, a leaf node is a variable that does not influence any other variables — it is typically an effect rather than a cause.
Which of the following variables is most likely to be a leaf node?
Only choose from the variables listed below. Do not answer with any variable not present in the list.

{chr(10).join(formatted_vars)}

Respond with only the variable name. Do not include any punctuation, whitespace, or explanation. 
Output just one variable name as plain text:""".strip()

    return prompt

def make_leaf_node_prompt2(variables: dict, data_description: str) -> str:
    active_nodes = [f"node{var}" for var in variables.keys()]
    variable_description_str = "\n".join([
        f'"node{var}": "{desc}"' for var, desc in variables.items()
    ])
    
    prompt = f"""You are an AI assistant tasked with identifying the most likely leaf node in a causal structure.
A leaf node is a variable that does not cause any other variables in the active set.  
Your goal is to determine the best leaf node among `active_nodes` using the given information.

Selection Criteria:
- A leaf node does not act as a cause for any other variable in `active_nodes`.
- If multiple candidates exist, select the one that is influenced by others but does not influence any other variable in active_nodes.
- Provide a concise reasoning for your selection.
- The leaf_node must be a single variable name from the active_nodes list.

Important Formatting Rules:
- Respond **only** with the variable name of the selected leaf node.
- Do **not** include any punctuation, reasoning, quotes, or formatting.
- Output **exactly one** variable name as plain text, matching one from the `active_nodes` list.
- Do **not** include any additional text before or after the variable name.

Example 1:
Input:
- Active Nodes: ["node_CloudCover", "node_Humidity", "node_Pressure", "node_Temperature"]
- Data Description: The dataset contains weather data recorded hourly with multiple atmospheric variables.
- Variable Descriptions:
[
"node_CloudCover": "The fraction of the sky covered by clouds.",
"node_Humidity”: “The amount of water vapor in the air.",
"node_Pressure”: “The atmospheric pressure at a given location.",
"node_Temperature”: “The measure of how hot or cold the air is."
]
Output: node_Temperature

Example 2:
Input:
- Active Nodes: ["node_WindSpeed", "node_Humidity", "node_Pressure", "node_Rainfall"s]
- Data Description: The dataset captures meteorological conditions over different seasons to analyze weather patterns.
- Variable Descriptions:
[
"node_WindSpeed": "The speed at which air moves.",
"node_Humidity": "The amount of water vapor in the air.",
"node_Pressure": "A measure of atmospheric force.",
"node_Rainfall": "The amount of precipitation occurring in a specific area."
]
Output: node_Rainfall

Active Nodes: {active_nodes}  
Data Description: {data_description}  
Variable Descriptions: 
[
{variable_description_str}
]  
Output: 
""".strip()
    return prompt


def make_leaf_node_prompt(version, variables, data_description=None):
    if version == "v1":
        return make_leaf_node_prompt1(variables)
    elif version == "v2":
        return make_leaf_node_prompt2(variables, data_description)
    else:
        raise ValueError(f"Unsupported prompt version: {version}")
        
# def generate(prompt, max_new_tokens=1):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]

#     # Generate text with logits output
#     with torch.no_grad():
#         output = model.generate(
#             input_ids,
#             do_sample=False,
#             max_new_tokens=max_new_tokens,  # Adjust as needed
#             output_logits=True,  # Enable output logits
#             return_dict_in_generate=True,
#             attention_mask=attention_mask, 
#             pad_token_id=tokenizer.eos_token_id 
#         )
        
#     return output

# def get_probs(prompt, variable, tokenizer, model, device): 
#     token_ids = tokenizer.encode(variable, add_special_tokens=False)
#     probs = []
#     log_probs = []

#     current_prompt = prompt
#     for i, token_id in enumerate(token_ids):
#         with torch.no_grad():
#             if i==0:
#                 output = generate(current_prompt, 2) #token0:" node", token1:"_{}"
#                 current_prompt += " node"
#                 current_prompt = current_prompt.strip()
#             else:
#                 output = generate(current_prompt, 1)
                
#         logits = torch.stack(output.logits, dim=1)
#         logits_vector = logits[0, -1, :]  
#         prob = F.softmax(logits_vector, dim=0)[token_id].cpu()
#         log_prob = F.log_softmax(logits_vector, dim=0)[token_id].cpu()
#         probs.append(prob)
#         log_probs.append(log_prob)

#         current_prompt += tokenizer.decode([token_id])
#         current_prompt = current_prompt.strip()
        
#     return probs, log_probs

def get_probs_vectorized(prompt, variable, tokenizer, model):
    device = model.device
    prefix = prompt + " node"
    prefix_enc = tokenizer(prefix, return_tensors="pt").to(device)
    var_enc = tokenizer(variable, add_special_tokens=False, return_tensors="pt").to(device)

    input_ids = torch.cat([prefix_enc.input_ids, var_enc.input_ids], dim=1)  
    attention_mask = torch.cat([prefix_enc.attention_mask, var_enc.attention_mask], dim=1)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits 

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  

    prefix_len = prefix_enc.input_ids.size(1)
    var_ids = var_enc.input_ids[0]            
    positions = torch.arange(prefix_len - 1, prefix_len - 1 + var_ids.size(0), device=device)

    token_log_probs = log_probs[0, positions, var_ids]  
    token_probs     = token_log_probs.exp()             

    return token_probs.cpu().tolist(), token_log_probs.cpu().tolist()


def analyze_log_probs(log_probs, alpha=0.5):
    lp_vals = [
        lp.item() if hasattr(lp, "item") else float(lp)
        for lp in log_probs
    ]
    joint_log_prob = sum(lp_vals)
    normalized_prob = torch.exp(torch.tensor(joint_log_prob / len(lp_vals))).item()
    normalized_prob2 = torch.exp(torch.tensor(alpha * joint_log_prob)).item()
    return {
        "joint_log_prob": joint_log_prob,
        "normalized_prob": normalized_prob,
        "normalized_prob2": normalized_prob2,
    }

def compute_min_pref_len_fast(token_ids):
    from collections import defaultdict
    prefix_counter = defaultdict(int)
    for var, ids in token_ids.items():
        for L in range(1, len(ids) + 1):
            prefix = tuple(ids[:L])
            prefix_counter[prefix] += 1

    min_pref_len = {}
    for var, ids in token_ids.items():
        for L in range(1, len(ids) + 1):
            prefix = tuple(ids[:L])
            if prefix_counter[prefix] == 1:
                min_pref_len[var] = L
                break
        else:
            min_pref_len[var] = len(ids)
    return min_pref_len

def get_node_joint_prob(model, tokenizer, prompt, variable_names, dataset_name, model_name="Llama3_1_8B_Instruct", prompt_version="v2", cache_dir="LLM_cache"):
    
    #cache
    model_cache_dir = os.path.join(cache_dir, model_name)
    os.makedirs(model_cache_dir, exist_ok=True)

    key_str = f"{dataset_name}_{prompt_version}_" + "_".join(sorted(variable_names))
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = os.path.join(model_cache_dir, f"{key_hash}.pkl")

    if os.path.exists(cache_path):
        print("📂 Loading LLM probs and normalized results from cache:", cache_path)
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
            return cache_data["normalized_results"]

    results = {}
    for v in variable_names:
        probs, log_probs = get_probs_vectorized(prompt, v, tokenizer, model)
        log_probs_tensor = torch.tensor([lp.item() if isinstance(lp, torch.Tensor) else lp for lp in log_probs])
        metrics = analyze_log_probs(log_probs_tensor, alpha=0.5) 
        
        results[v] = {
            "token_probs": [p.item() if isinstance(p, torch.Tensor) else p for p in probs],
            "log_probs": [lp.item() if isinstance(lp, torch.Tensor) else lp for lp in log_probs],
            **metrics
        }
    
    #normalizeld probability
    norm1 = {k: v["normalized_prob"] for k, v in results.items()}
    norm2 = {k: v["normalized_prob2"] for k, v in results.items()}
    
    sum_norm1 = sum(norm1.values())
    sum_norm2 = sum(norm2.values())
    
    normalized_results = {
    var: {
            "normalized_prob": norm1[var] / sum_norm1,
            "normalized_prob2": norm2[var] / sum_norm2 
        }
        for var in results
    }

    raw_results = {
        var: {
            "token_probs": results[var]["token_probs"],
            "log_probs": results[var]["log_probs"]
        }
        for var in results
    }

    cache_data = {
        "normalized_results": normalized_results,
        "raw_results": raw_results
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    return normalized_results
    

def get_data_prob(order, active_nodes, dataset_name, num_variances=30, cache_dir="ensemble_cache"):

    #cache
    os.makedirs(cache_dir, exist_ok=True)
    sub_cache_dir = os.path.join(cache_dir, dataset_name)
    os.makedirs(sub_cache_dir, exist_ok=True)

    key_str = f"{dataset_name}/{dataset_name}_{order}_{active_nodes}_{num_variances}"
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = os.path.join(sub_cache_dir, f"{key_hash}.pkl")

    if os.path.exists(cache_path):
        print("📂 Loading SciNO cached results from:", cache_path)
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        return cache_data["combined_result"]

    ensemble_variance_list, variance_list, node_variance_means, node_variance_stds, leaf = ensemble_topological_ordering(order, active_nodes, num_model=num_variances)

    all_nodes = set()
    for r in ensemble_variance_list:
        all_nodes.update(r.keys())

    rank_sum = {node: 0 for node in all_nodes}
    ci_overlap_count = {node: 0 for node in all_nodes}
    num_rounds = len(ensemble_variance_list)

    z = 1.96  # 95% confidence

    ci_bounds = {
        node: (
            node_variance_means[node] - z * (node_variance_stds[node] / np.sqrt(num_rounds)),  # lower
            node_variance_means[node] + z * (node_variance_stds[node] / np.sqrt(num_rounds))   # upper
        )
        for node in node_variance_means
    }
    
    for ensemble_result in ensemble_variance_list:
        min_node = min(ensemble_result, key=ensemble_result.get)

        # ci_overlap_prob
        _, min_ci_upper = ci_bounds[min_node]
        for node, var in ensemble_result.items():
            ci_lower, _ = ci_bounds[node]
            if ci_lower <= min_ci_upper:
                ci_overlap_count[node] += 1

        # rank_prob
        sorted_nodes = sorted(ensemble_result.items(), key=lambda x: x[1])
        for rank, (node, _) in enumerate(sorted_nodes):
            rank_sum[node] += rank

    P_ci_overlap_count = {node: count / num_rounds for node, count in ci_overlap_count.items()}  

    avg_rank = {node: r / num_rounds for node, r in rank_sum.items()}
    exp_scores = {node: np.exp(-avg_rank[node]) for node in avg_rank}
    Z = sum(exp_scores.values())
    P_rank = {node: score / Z for node, score in exp_scores.items()} # P_rank


    combined_result = {
        node: {
            "ci_overlap_prob": P_ci_overlap_count.get(node, 0),
            "rank_prob": P_rank.get(node, 0),
        }
        for node in sorted(set(P_rank) | set(P_ci_overlap_count))
    }

    ensemble_raw_results = {
        "ensemble_variance_list": ensemble_variance_list,
        "node_variance_means": node_variance_means,
        "node_variance_stds": node_variance_stds,
        "leaf": leaf
    }
    cache_data = {
        "combined_result": combined_result,
        "ensemble_raw_results": ensemble_raw_results
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    return combined_result


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


def main():
    load_dotenv()

    CONFIG_PATH = 'configs/control/_control.yml'
    DESCRIPTION_PATH = "configs/control/description.json"

    with open(CONFIG_PATH, "r") as f: 
        configs = yaml.safe_load(f)
    config = dict2namespace(configs)

    with open(DESCRIPTION_PATH, "r") as f:
        variable_list = json.load(f)

    model_name = config.model_name
    auth_token = os.getenv("HUGGINGFACE_TOKEN")

    # Load the tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(model_name, token=auth_token, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token, force_download=True, trust_remote_code=True)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_name = config.dataset_name 
    control = config.control
    scino_prob_list = config.scino_prob_list
    llm_prob_list =  config.llm_prob_list
    num_variances = config.num_variances
    prompt_version = config.prompt_version


    if control:
        prob_combinations = [(d, l) for d in scino_prob_list for l in llm_prob_list]
    else:
        prob_combinations = [("none", c) for c in llm_prob_list]  


    for scino_prob, llm_prob in prob_combinations:
    

        log_suffix = f"{dataset_name}_{scino_prob}_{llm_prob}".replace("none", "llmonly")
        stdout_filename = f"logs/stdout_{log_suffix}_test.log"
        result_filename = f"logs/result_{dataset_name}_test.jsonl"

        os.makedirs("logs", exist_ok=True)
        logfile = open(stdout_filename, "a", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, logfile) 

        print(f"\n=== Running {dataset_name} | {scino_prob=} | {llm_prob=} ===\n")
        
        variables = variable_list[dataset_name]["variables"].copy()
        data_description = variable_list[dataset_name]["data_description"]
        ordered_variables = []

        # Load dataset
        true_causal_matrix, _, _ = get_dataset(dataset_name)
        to_index, to_name = mapping_fun(dataset_name)

        while len(variables) > 1:
            order = [get_mapped_index(var, dataset_name, to_index) for var in ordered_variables]
            active_nodes = [get_mapped_index(var, dataset_name, to_index) for var in variables]
            print("\norder:", order)
            print("active_nodes:", active_nodes)

            prompt = make_leaf_node_prompt(prompt_version, variables, data_description)
            normalized_results = get_node_joint_prob( 
                model = model,
                tokenizer = tokenizer,
                prompt = prompt, 
                variable_names=variables.keys(), 
                dataset_name=dataset_name,  
                model_name=model_name, 
                prompt_version=prompt_version
                ) 

            if control:
                scino_results = get_data_prob(
                    order, 
                    active_nodes, 
                    dataset_name = dataset_name, 
                    num_variances = num_variances
                    ) 
                named_scino_results = {
                    get_mapped_name(idx, dataset_name, to_name): values
                    for idx, values in scino_results.items()
                }
                print("LLM_results:", normalized_results)
                print("SciNO:", named_scino_results)

                combined_scores = {
                    var: named_scino_results[var][scino_prob] * normalized_results[var][llm_prob]
                    for var in named_scino_results
                }

                print("Combined_scores:", combined_scores)
                leaf = max(combined_scores, key=combined_scores.get)
                
            else:
                print("LLM_results:", normalized_results)
                leaf = max(variables, key=lambda var: normalized_results[var][llm_prob])


            ordered_variables.append(leaf)
            del variables[leaf]

        last_var = next(iter(variables))
        ordered_variables.append(last_var)
        ordered_variables.reverse()

        order_index = [get_mapped_index(var, dataset_name, to_index) for var in ordered_variables]
        print(order_index)
        errors = num_errors(order_index, true_causal_matrix)
        print("\nOrder Divergence:",errors)

        # Save results to a log file
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "errors": errors,
            "llm_prob": llm_prob,
            "scino_prob": None if scino_prob == "none" else scino_prob,
            "control": control,
            "num_variances": num_variances,
            "ordered_variables": ordered_variables,
            "order_index": order_index
        }

        log_data = convert_to_builtin(log_data)

        with open(result_filename, "a") as logfile:
            logfile.write(json.dumps(log_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()