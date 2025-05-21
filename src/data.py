import os

import numpy as np
import pandas as pd
import torch
import networkx as nx

import cdt


def get_dataset(dataset, n_node=None, graph_idx=None, num_samples=1000):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node_names = None
    
    if dataset == "synthetic":
        output_dir = os.path.join(f'data/ER/n_nodes_{n_node}')
        X_path = os.path.join(output_dir, f'X_{graph_idx}_ver1.npy')
        X = np.load(X_path)
        X = (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True) / 3
        causal_matrix_path = os.path.join(output_dir, f'true_causal_matrix_{graph_idx}_ver1.npy')
        true_causal_matrix = np.load(causal_matrix_path)

    elif dataset in ['ecoli70','magic-irri', 'magic-niab', 'arth150']:
        output_dir = os.path.join(f'data/bnlearn/{dataset}')
        X_path = os.path.join(output_dir, f'X.npy')
        X = np.load(X_path)
        causal_matrix_path = os.path.join(output_dir, f'true_causal_matrix.npy')
        true_causal_matrix = np.load(causal_matrix_path)
        csv_path = os.path.join(output_dir, f'true_causal_matrix.csv')
        df = pd.read_csv(csv_path)
        node_names = df.columns
    
    elif dataset in ['ecoli70_linear','magic-irri_linear', 'magic-niab_linear', 'arth150_linear']:
        output_dir = os.path.join(f'data/bnlearn/{dataset.split("_")[0]}')
        X_path = os.path.join(output_dir, f'X_linear.npy')
        X = np.load(X_path)
        causal_matrix_path = os.path.join(output_dir, f'true_causal_matrix.npy')
        true_causal_matrix = np.load(causal_matrix_path)

    elif dataset in ["sachs", "sachs_nocycle"]:
        data, solution = cdt.data.load_dataset('sachs')
        node_names = list(data.columns)
        true_causal_matrix = nx.to_numpy_array(solution).T
        if dataset == "sachs_nocycle":
            remove_vars = ["praf", "plcg", "PIP2"]
            remove_indices = [data.columns.get_loc(var) for var in remove_vars]
            true_causal_matrix = np.delete(true_causal_matrix, remove_indices, axis=0)
            true_causal_matrix = np.delete(true_causal_matrix, remove_indices, axis=1)
            node_names = [name for i, name in enumerate(node_names) if i not in remove_indices]
            data = data.drop(columns=remove_vars)
        X = data.values

    elif dataset == "physics":
        data = pd.read_csv("data/physics_generation/physics_7nodes_5000.csv")
        node_names = list(data.columns)
        X = data.values
        true_causal_df = pd.read_csv("data/physics_generation/physics_truth/physics_7node_truth.csv", index_col=0)
        true_causal_matrix = true_causal_df.values

    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")

    X = torch.from_numpy(X).float().to(device)
    return true_causal_matrix, X[:num_samples], node_names