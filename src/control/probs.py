import torch
import numpy as np
import torch.nn.functional as F
import hashlib
import os
import pickle
from src.control.ordering import ensemble_topological_ordering

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

def get_node_joint_prob(model, tokenizer, prompt, variable_names, dataset_name, model_name="Llama3_1_8B_Instruct", cache_dir="LLM_cache"):
    
    #cache
    model_cache_dir = os.path.join(cache_dir, model_name)
    os.makedirs(model_cache_dir, exist_ok=True)

    key_str = f"{dataset_name}_" + "_".join(sorted(variable_names))
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

# ========================================================================================

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