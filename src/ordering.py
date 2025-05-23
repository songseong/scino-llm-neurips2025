import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.func import vmap, jacrev, jacfwd
from torch.utils.data import DataLoader

def topological_ordering(model, X, n_nodes, device, batch_size, config, step=None):
    eval_batch_size = config.evaluation.eval_batch_size
    model.eval()
    order = []
    active_nodes = list(range(n_nodes))

    random_sampling = getattr(config.evaluation, "random_sampling", False)
    
    steps_list = np.linspace(0, config.evaluation.t_max, config.evaluation.n_votes + 1)
    if not config.evaluation.masking and not config.evaluation.residue:
        steps_list = [config.training.diffan.n_steps // 2]

    pbar = tqdm(range(n_nodes - 1), desc="Nodes ordered")
    for _ in pbar:
        if random_sampling:
            indices = random.sample(range(len(X)), batch_size)
            sampled_X = X[indices]
        else:
            sampled_X = X[:batch_size]

        data_loader = DataLoader(sampled_X, eval_batch_size, drop_last=False, shuffle=False)

        leaves = []
        for steps in steps_list:
            model_fn_functorch = get_model_function_with_residue(model, steps, active_nodes, order, config, device)
            leaf_ = compute_jacobian_and_get_leaf(model_fn_functorch, data_loader, active_nodes, device, config.evaluation.masking, config)
            if not config.evaluation.masking and not config.evaluation.residue:
                order = leaf_.tolist()
                order.reverse()
                return order
            leaves.append(leaf_)
        most_common_leaf = Counter(leaves).most_common(1)[0][0]
        leaf_global = active_nodes[most_common_leaf]
        order.append(leaf_global)
        active_nodes.pop(most_common_leaf)

    order.append(active_nodes[0])
    order.reverse()
    return order


def get_model_function_with_residue(model, step, active_nodes, order, config, device):
    t_functorch = (torch.ones((1,)) * step).to(device).float()
    get_score_active = lambda x: model(x, t_functorch)[:, active_nodes]
    get_score_previous_leaves = lambda x: model(x, t_functorch)[:, order]
    
    def model_fn_functorch(X):
        score_active = get_score_active(X)
        if config.evaluation.residue and len(order) > 0:
            score_previous_leaves = get_score_previous_leaves(X).squeeze()
            jacobian_ = jacfwd(get_score_previous_leaves)(X).squeeze()
            if len(order) == 1:
                jacobian_, score_previous_leaves = jacobian_.unsqueeze(0), score_previous_leaves.unsqueeze(0)

            with torch.no_grad():
                score_active += torch.einsum("i,ij -> j", score_previous_leaves / jacobian_[:, order].diag(),
                                             jacobian_[:, active_nodes])
        return score_active
    return model_fn_functorch


def compute_jacobian_and_get_leaf(model_fn_functorch, data_loader, active_nodes, device, masking, config):
    jacobian = []
    for x_batch in data_loader:
        if masking:
            x_batch = get_masked(x_batch, active_nodes, device)
        jacobian_ = vmap(jacrev(model_fn_functorch))(x_batch.unsqueeze(1)).squeeze()
        jacobian.append(jacobian_[..., active_nodes].detach().cpu().numpy())
    jacobian = np.concatenate(jacobian, 0)
    #### CaPS 
    if config.CaPS:
        jacobian.diagonal().mean(axis=0)
        jacobian_mean_diag = jacobian.diagonal().mean(axis=0)
        var_sorted_nodes = np.argsort(jacobian_mean_diag)
        leaf = var_sorted_nodes[-1]
    #### DiffAN
    else:
        leaf = get_leaf(jacobian)
    return leaf


def get_masked(x, active_nodes, device):
    dropout_mask = torch.zeros_like(x).to(device)
    dropout_mask[:, active_nodes] = 1
    return (x * dropout_mask).float()

def get_leaf(jacobian_active):
    jacobian_var_diag = jacobian_active.var(0).diagonal()

    var_sorted_nodes = np.argsort(jacobian_var_diag)
    return var_sorted_nodes[0]