from transformers import AutoTokenizer, AutoModelForCausalLM
from src.control.mapping import mapping_fun, get_mapped_index, get_mapped_name
from src.utils import num_errors, dict2namespace
from src.data import get_dataset
from datetime import datetime
import json
import sys
import torch
import yaml
import os
from dotenv import load_dotenv
from src.control.utils import Tee, convert_to_builtin, make_leaf_node_prompt, softmax
from src.control.probs import get_node_joint_prob, get_data_prob

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
    partial_context = config.partial_context
    tau = config.tau

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

            prompt = make_leaf_node_prompt(variables, data_description)
            normalized_results = get_node_joint_prob( 
                model = model,
                tokenizer = tokenizer,
                prompt = prompt, 
                variable_names=variables.keys(), 
                dataset_name=dataset_name,  
                model_name=model_name
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

                if partial_context:
                    combined_scores = {}
                    supervised_vars = [var for var in named_scino_results if variables.get(var).strip() != ""]

                    if supervised_vars:
                        scino_probs = [named_scino_results[var][scino_prob] for var in supervised_vars]
                        full_scino_probs = [named_scino_results[var][scino_prob] for var in named_scino_results]
                        softmax_probs = softmax(scino_probs, full_scino_probs, tau)

                        for var, prob in zip(supervised_vars, softmax_probs):
                            combined_scores[var] = prob * normalized_results[var][llm_prob]

                    default_score = 1 / len(active_nodes) if active_nodes else 0
                    for var in named_scino_results:
                        if variables.get(var).strip() == "":
                            combined_scores[var] = named_scino_results[var][scino_prob] * default_score

                else:
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