import numpy as np
import uuid
import pandas as pd
import os 
import json
import argparse

from argparse import Namespace
from types import SimpleNamespace

def num_errors(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err

def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def np_to_csv(array, save_path):
    id = str(uuid.uuid4())
    output = os.path.join(save_path, 'tmp_' + id + '.csv')
    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)
    return output

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def namespace2dict(obj):
    if isinstance(obj, (Namespace, SimpleNamespace)):
        return {k: namespace2dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: namespace2dict(v) for k, v in obj.items()}
    else:
        return obj

def save_num_errors(dict_errors, n_nodes, output_dir):
    converted_dict = {key: {int(k): int(v) for k, v in value.items()} for key, value in dict_errors.items()}
    
    file_name = f"n{n_nodes}_errors.json"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as f:
        json.dump(converted_dict, f, indent=4)
    print(f"Errors saved to {file_path}")

def save_num_errors2(dict_errors, n_nodes, output_dir, graph_idx, ordering_option = None):
    converted_dict = {key: {int(k): int(v) for k, v in value.items()} for key, value in dict_errors.items()}
    
    s_idx, e_idx = graph_idx
    file_name = f"n{n_nodes}_errors_g{s_idx}_{e_idx}" + (f"_{ordering_option}.json" if ordering_option else ".json")
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as f:
        json.dump(converted_dict, f, indent=4)
    print(f"Errors saved to {file_path}")

def compute_rbf_kernel(X, n_kernels=5, mul_factor=2.0, bandwidth=None):
    """
    Computes a multi-scale RBF kernel matrix from [Gretton et al. 2012].

    Args:
        X: (n_samples, n_features)
        n_kernels: Number of RBF kernels
        mul_factor: Scaling factor between bandwidths
        bandwidth: Optional fixed base bandwidth. If None, use mean L2.

    Returns:
        (n_samples, n_samples) kernel matrix
    """
    L2 = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2, axis=1).reshape(1, -1) - 2 * np.dot(X, X.T)

    if bandwidth is None:
        bandwidth = np.sum(L2) / (X.shape[0] ** 2 - X.shape[0])

    multipliers = mul_factor ** (np.arange(n_kernels) - n_kernels // 2)
    kernels = np.exp(-L2[None, :, :] / (bandwidth * multipliers[:, None, None]))
    return np.sum(kernels, axis=0)

def compute_mmd(x, y, n_kernels=5, mul_factor=2.0, bandwidth=None):
    """
    Biased MMD estimate using multi-scale RBF kernel.
    https://github.com/yiftachbeer/mmd_loss_pytorch?tab=readme-ov-file

    Args:
        x, y: jnp.ndarray, shape (n_samples, n_features)
        n_kernels: Number of RBF kernels
        mul_factor: Kernel bandwidth multiplier
        bandwidth: Base bandwidth. If None, uses mean heuristic.

    Returns:
        float: MMD score
    """
    X = np.vstack([x, y])
    K = compute_rbf_kernel(X, n_kernels=n_kernels, mul_factor=mul_factor, bandwidth=bandwidth)

    n = x.shape[0]
    K_xx = K[:n, :n]
    K_yy = K[n:, n:]
    K_xy = K[:n, n:]

    mmd = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
    return mmd