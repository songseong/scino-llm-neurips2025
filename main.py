import torch
import numpy as np
import pandas as pd
import os
import yaml
import argparse
import wandb 
import json

from datetime import datetime
import shutil
from cdt.metrics import SID
from src.model import Model, ModelProbe
from src.utils import num_errors, dict2namespace, namespace2dict
from src.data import get_dataset

np.set_printoptions(precision=3)
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)  

def main():
    
    parser = argparse.ArgumentParser(description='SciNO')
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument('--models', nargs='+', type=str, required=True, choices=['DiffAN', 'FNO'], help='Select one models to run')
    parser.add_argument('--load-ckpt', action='store_true', help='Load model from a checkpoint')
    parser.add_argument('--CaPS', action='store_true', help='Using CaPS method')
    parser.add_argument('--probe', action='store_true', help='linear probing')
    parser.add_argument('--pretrain', action='store_true', help='pretrain')

    args = parser.parse_args()
    config_file_path = os.path.join("configs/", args.config)
    
    with open(config_file_path, "r") as f: 
        configs = yaml.safe_load(f)

    wandb.login()

    config = dict2namespace(configs)
    config.config_name = os.path.splitext(os.path.basename(args.config))[0]
    print(config.config_name)
    config.load_ckpt = args.load_ckpt
    config.CaPS = args.CaPS
    config.probe = args.probe
    config.config_file = config_file_path
    config.use_wandb = config.setup.use_wandb
    config.pretrain = args.pretrain

    n_nodes = config.datasets.n_nodes
    num_graphs_list = config.datasets.num_graphs_list
    output_path = config.setup.output_path

    #model save settings
    experiment_name = config.setup.experiment_name 
    if experiment_name is None:
        experiment_name = datetime.now().strftime('%m%d_%H%M%S')  
    print(experiment_name)

    config.folder_path = os.path.join(config.setup.save_folder, experiment_name)

    if (config.training.model_save) & (not config.load_ckpt):
        os.makedirs(config.folder_path, exist_ok=True)
        shutil.copy(config.config_file, config.folder_path)

    dataset_names = ['ecoli70_linear','magic-irri_linear', 'magic-niab_linear', 'arth150_linear', 'sachs_nocycle', 'sachs_bn', 'sachs', 'dream', 'asia', 'cancer', 'earthquake', 'survey', 'child', 'physics', 'ecoli70', 'magic-niab', 'magic-irri', 'arth150']
    dataset = next((ds for ds in dataset_names if ds in config.config_name), 'synthetic')
    dataset = dataset.strip()
    config.dataset_name = dataset
    print("="*20)
    print(dataset)
    print("="*20)

    def test_two_model(config, n_nodes, num_graphs):

        final_results = {}
        for model_name in args.models:
            final_results[f"{model_name}_mean"], final_results[f"{model_name}_std"]  = [], []
            if config.evaluation.pruning_P:
                final_results[f"{model_name}_SID_P_mean"], final_results[f"{model_name}_SID_P_std"] = [], []

        output_dir = f"../ckpt/{config.setup.experiment_name}"

        s_idx = getattr(config.datasets, "graph_idx", (0, num_graphs - 1))[0]
        e_idx = getattr(config.datasets, "graph_idx", (0, num_graphs - 1))[1]
        graph_range = range(s_idx, e_idx + 1) 

        ordering_option = getattr(config.evaluation, "ordering_option", None)

        for n_node in n_nodes:

            wandb.init(project='SciNO',
                       config = configs,
                       group = f'n_node: {n_node}',
                       name = experiment_name)

            results = {"DiffAN": {"error": [], "SID_R": [], "SID_P": []}, 
                       "FNO": {"error": [], "SID_R": [], "SID_P": []}}
            dict_errors = {model_name: {} for model_name in args.models}  

            for graph_idx in graph_range:
                config.training.graph_idx = graph_idx
                config.training.n_node = n_node

                true_causal_matrix, X, _ = get_dataset(dataset, n_node, graph_idx, config.datasets.num_samples)

                model_type_map = {"DiffAN": "mlp", "FNO": "fno"}
                model_dict = {
                    name: Model(config, n_nodes=n_node, model_type=model_type_map[name])
                    for name in args.models
                    }

                for model_name in args.models:
                    if args.probe and (model_name != "DiffAN"):
                        model = ModelProbe(config, n_nodes=n_node, model_type='fno')

                        torch.cuda.reset_peak_memory_stats()

                        adj_matrix, order = model.probing(X, true_causal_matrix)
                        print("predicted order: ", order)
                        
                        # best_results.json path
                        best_result_path = "prob_results/best_results.json"

                        if os.path.exists(best_result_path):
                            with open(best_result_path, "r") as f:
                                best_results = json.load(f)
                        else:
                            best_results = {}
                        cur_errors = num_errors(order, true_causal_matrix)
                        # 현재 dataset의 best result 갱신 조건 확인
                        if (dataset not in best_results) or (cur_errors < best_results[dataset]["num_errors"]):
                            best_results[dataset] = {
                                "num_errors": int(cur_errors),
                                "order": order,
                                "config": namespace2dict(config)
                            }
                            print(f"✅ Best result updated for {dataset} with {cur_errors} errors.")
                        else:
                            print(f"ℹ️ No update. Current num_errors ({cur_errors}) is not better than best ({best_results[dataset]['num_errors']}).")

                        # 저장
                        with open(best_result_path, "w") as f:
                            json.dump(best_results, f, indent=2)
                    else:
                        model = model_dict[model_name]

                        adj_matrix, order = model.fit(X, true_causal_matrix)

                    errors = num_errors(order, true_causal_matrix)
                    print(f"{model_name} Num errors {errors}")

                    dict_errors[model_name][graph_idx] = errors
                    
                    results[model_name]["error"].append(errors)
                    if config.evaluation.pruning_P:
                        results[model_name]["SID_P"].append(SID(true_causal_matrix, adj_matrix).item())     

            for model_name in args.models:
                errors_array = np.array(results[model_name]["error"])
                final_results[f'{model_name}_mean'].append(np.mean(errors_array))
                final_results[f'{model_name}_std'].append(np.std(errors_array))
                if config.evaluation.pruning_P:
                    SID_P_array = np.array(results[model_name]["SID_P"])
                    final_results[f'{model_name}_SID_P_mean'].append(np.mean(SID_P_array))
                    final_results[f'{model_name}_SID_P_std'].append(np.std(SID_P_array))
                
        return final_results

    os.makedirs("diffan_results", exist_ok=True)
    output_path = os.path.join("diffan_results", output_path)
    
    with open(output_path, "a") as file:
        for num_graphs in num_graphs_list:
            results_dic = test_two_model(config, n_nodes, num_graphs)

            df = pd.DataFrame(results_dic, index=n_nodes)
            print(df)
            file.write(f"<{config.setup.experiment_name}>\n")
            file.write(f"{num_graphs} graphs\n")
            file.write(f"{df}\n")


if __name__ == "__main__":
    main()