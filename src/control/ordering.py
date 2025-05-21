import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.func import vmap, jacrev, jacfwd
from torch.utils.data import DataLoader

import yaml
from src.utils import dict2namespace
from src.data import get_dataset
from src.fno_LFPE import DiffFNO_LFPE

# Static paths
CONFIG_PATH = 'configs/control/_control.yml'

with open(CONFIG_PATH, "r") as f: 
        configs = yaml.safe_load(f)
config = dict2namespace(configs)

data_config_path = f"configs/control/_control_{config.dataset_name}.yml"
with open(data_config_path, "r") as f: 
    configs = yaml.safe_load(f)
dataset_config = dict2namespace(configs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def restructure_variances(variance_list):
    node_variance_lists = {}
    for var_dict in variance_list:
        for node_id, var in var_dict.items():
            if node_id not in node_variance_lists:
                node_variance_lists[node_id] = []
            node_variance_lists[node_id].append(var)

    node_variance_means = {
        node_id: np.mean(var_list) for node_id, var_list in node_variance_lists.items()
    }
    node_variance_stds = {
        node_id: np.std(var_list) for node_id, var_list in node_variance_lists.items()
    }
    
    leaf = min(node_variance_means, key=node_variance_means.get)

    return variance_list, node_variance_lists, node_variance_means, node_variance_stds, leaf

def ensemble_topological_ordering(order, active_nodes, num_model=30):
    base_path = f"ckpt/_ckpt_control/_fno_{dataset_config.dataset_name}"
    n_nodes = dataset_config.n_nodes

    all_variances = []
    for m_idx in tqdm(range(num_model)):
        model_path = f"{base_path}/fno_n{n_nodes}_g{m_idx}.pth"
        try:
            leaf, variance_dict = topological_ordering_llm(order, active_nodes, model_path=model_path)
            all_variances.append(variance_dict)
        except Exception as e:
            print(f"⚠️ Skipping model {m_idx} due to error: {e}")
    
    return restructure_variances(all_variances)

def initialize_model_data(config, dataset_config, model_path=None):
    dataset = dataset_config.dataset_name
    _, X, node_names = get_dataset(dataset, dataset_config.evaluation.ordering_batch_size) 
    
    n_fourier_layers = getattr(dataset_config.model, "n_fourier_layers", None)

    model = DiffFNO_LFPE(dataset_config.n_nodes, n_fourier_layers=n_fourier_layers).to(device)
    ckpt = torch.load(model_path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch_size = dataset_config.evaluation.ordering_batch_size
    random_sampling = getattr(dataset_config.evaluation, "random_sampling", False)
    if random_sampling:
        indices = random.sample(range(len(X)), batch_size)
        sampled_X = X[indices]
    else:
        sampled_X = X[:batch_size]

    return model, sampled_X
  

def topological_ordering_llm(order, active_nodes, data_loader=None, model_path=None): 

    eval_batch_size = dataset_config.evaluation.eval_batch_size
    model, sampled_X = initialize_model_data(config, dataset_config, model_path)
    
    if data_loader==None:
        data_loader = DataLoader(sampled_X, eval_batch_size, drop_last=True)

    steps_list = np.linspace(0, dataset_config.evaluation.t_max, dataset_config.evaluation.n_votes + 1)

    leaves, variance_list = [], []
    for steps in steps_list:
        model_fn_functorch = get_model_function_with_residue2(model, steps, active_nodes, order, config, device)
        leaf_, variance_ = compute_jacobian_and_get_leaf2(model_fn_functorch, data_loader, active_nodes, device, dataset_config.evaluation.masking)
        leaves.append(leaf_)
        variance_list.append(variance_)
    most_common_leaf = Counter(leaves).most_common(1)[0][0]
    leaf_global = active_nodes[most_common_leaf]

    avg_variance = np.mean(variance_list, axis=0)  
    variance_dict = {active_nodes[i]: avg_variance[i] for i in range(len(active_nodes))}

    return leaf_global, variance_dict

def get_model_function_with_residue2(model, step, active_nodes, order, config, device):
    t_functorch = (torch.ones((1,)) * step).to(device).float()
    get_score_active = lambda x: model(x, t_functorch)[:, active_nodes]
    get_score_previous_leaves = lambda x: model(x, t_functorch)[:, order]
    
    def model_fn_functorch(X):
        score_active = get_score_active(X)
        if dataset_config.evaluation.residue and len(order) > 0:
            score_previous_leaves = get_score_previous_leaves(X).squeeze()
            jacobian_ = jacfwd(get_score_previous_leaves)(X).squeeze()
            if len(order) == 1:
                jacobian_, score_previous_leaves = jacobian_.unsqueeze(0), score_previous_leaves.unsqueeze(0)

            with torch.no_grad():
                score_active += torch.einsum("i,ij -> j", score_previous_leaves / jacobian_[:, order].diag(),
                                             jacobian_[:, active_nodes])
        return score_active
    return model_fn_functorch

def compute_jacobian_and_get_leaf2(model_fn_functorch, data_loader, active_nodes, device, masking):
    jacobian = []
    for x_batch in data_loader:
        if masking:
            x_batch = get_masked(x_batch, active_nodes, device)
        jacobian_ = vmap(jacrev(model_fn_functorch))(x_batch.unsqueeze(1)).squeeze()
        jacobian.append(jacobian_[..., active_nodes].detach().cpu().numpy())
    jacobian = np.concatenate(jacobian, 0)
    leaf, variance = get_leaf2(jacobian)
    return leaf, variance

def get_masked(x, active_nodes, device):
    dropout_mask = torch.zeros_like(x).to(device)
    dropout_mask[:, active_nodes] = 1
    return (x * dropout_mask).float()

def get_leaf2(jacobian_active):
    jacobian_var_diag = jacobian_active.var(0).diagonal()
    var_sorted_nodes = np.argsort(jacobian_var_diag)
    return var_sorted_nodes[0], jacobian_var_diag

