import argparse
import os
import random
import torch
import pandas as pd
import numpy as np

import pandas as pd
import time
from castle.metrics import MetricsDAG

from cdt.metrics import SID

from src.caps.utils import *
from pathlib import Path
from cdt.utils.R import launch_R_script
import tempfile
import json
from src.data import get_dataset

torch.set_printoptions(linewidth=1000)
np.set_printoptions(linewidth=1000)
def blue(x): return '\033[94m' + x + '\033[0m'
def red(x): return '\033[31m' + x + '\033[0m'

only_order = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', nargs='?', const=False, type=int, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default='CaPS')
    parser.add_argument('--lambda1', type=float, default=50.0)
    parser.add_argument('--lambda2', type=float, default=50.0)

    parser.add_argument('--dataset', type=str, default='sachs')
    parser.add_argument('--norm', type=bool, default=False)
    parser.add_argument('--num_nodes', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--linear_sem_type', type=str, default='gauss')
    parser.add_argument('--nonlinear_sem_type', type=str, default='gp')
    parser.add_argument('--linear_rate', type=float, default=1.0)

    parser.add_argument('--manualSeed', type=str, default='False')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--method', type=str, default='stein')

    args = parser.parse_args()
    args.manualSeed = True if args.manualSeed == 'True' else False
    return args

def order_divergence(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err

def evaluate(args, dag, GT_DAG):
    print('pred_dag:\n', dag)
    print('gt_dag:\n', GT_DAG.astype(int))
    print(blue('edge_num: ' + str(np.sum(dag))))
    mt = MetricsDAG(dag, GT_DAG)
    sid = SID(GT_DAG, dag)
    mt.metrics['sid'] = sid.item()
    print(blue(str(mt.metrics)))
    return mt.metrics

def cam_pruning(A, X, cutoff):
    with tempfile.TemporaryDirectory() as save_path:
        pruning_dir = Path(__file__).parent / "src/caps"
        pruning_path = pruning_dir / "pruning_R_files/cam_pruning.R"  

        prev_dir = os.getcwd()                 
        os.chdir(pruning_dir)
        
        data_np = np.array(X.detach().cpu().numpy())
        data_csv_path = np_to_csv(data_np, save_path)
        dag_csv_path = np_to_csv(A, save_path)

        arguments = dict()
        arguments['{PATH_DATA}'] = data_csv_path
        arguments['{PATH_DAG}'] = dag_csv_path
        arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
        arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
        arguments['{CUTOFF}'] = str(cutoff)
        arguments['{VERBOSE}'] = "FALSE" # TRUE, FALSE


        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A
        
        dag = launch_R_script(str(pruning_path), arguments, output_function=retrieve_result)

        os.chdir(prev_dir)

    return dag, adj2order(dag)


def train_test(args, train_set, GT_DAG, data_ls, runs):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.model == 'CaPS':
        
        def Stein_hess(X, eta_G, eta_H, s = None):
            """
            Estimates the diagonal of the Hessian of log p_X at the provided samples points
            X, using first and second-order Stein identities
            """
            n, d = X.shape
            X = X.to(device)
            X_diff = X.unsqueeze(1)-X
            if s is None:
                D = torch.norm(X_diff, dim=2, p=2)
                s = D.flatten().median()
            K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
            
            nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
            G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n).to(device)), nablaK)
            
            if args.batch_size:
                nabla2K_parts = []

                batch_size = args.batch_size
                for i in range(0, n, batch_size):
                    X_batch = X[i:i+batch_size]
                    X_diff_batch = X_batch.unsqueeze(1) - X  # shape (B, n, d)
                    K_batch = torch.exp(-torch.norm(X_diff_batch, dim=2)**2 / (2 * s**2)) / s
                    part = torch.einsum('kij,ki->kj', -1/s**2 + X_diff_batch**2/s**4, K_batch)
                    nabla2K_parts.append(part)
                nabla2K = torch.cat(nabla2K_parts, dim=0)
                
            else: 
                nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
            
            return (-G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n).to(device)), nabla2K)).to('cpu')

        def compute_top_order(X, eta_G, eta_H, dispersion="mean"):
            n, d = X.shape
            order = []
            active_nodes = list(range(d))
            for i in range(d-1):
                H = Stein_hess(X, eta_G, eta_H)
                if dispersion == "mean": # Lemma 1 of CaPS
                    l = int(H.mean(axis=0).argmax())
                    # l = int(H.var(axis=0).argmin())
                else:
                    raise Exception("Unknown dispersion criterion")

                order.append(active_nodes[l])
                active_nodes.pop(l)

                X = torch.hstack([X[:,0:l], X[:,l+1:]])
            order.append(active_nodes[0])
            order.reverse()

            # compute parents score
            # active_nodes = list(range(d))
            # full_H = Stein_hess(full_X, eta_G, eta_H).mean(axis=0)
            # parents_score = np.zeros((d,d))
            # for i in range(d):
            #     curr_X = torch.hstack([full_X[:,0:i], full_X[:,i+1:]])
            #     curr_H = Stein_hess(curr_X, eta_G, eta_H).mean(axis=0)
            #     parents_score[i] = get_parents_score(curr_H, full_H, i)
            # print(parents_score)
            
            parents_score = None

            return order, parents_score
        
        cutoff = 0.001
        order, parents_score = compute_top_order(train_set, eta_G=0.001, eta_H=0.001, dispersion="mean")

        print(blue(str(order)))

    if only_order: 
        return None, order
    else:
        init_dag = full_DAG(order)
        
        dag, _ = cam_pruning(init_dag, train_set, cutoff)

    return evaluate(args, dag, GT_DAG), order


if __name__ == '__main__':
    args = get_args()
    if only_order:
        metrics_list_dict = {'D_top':[]}
    else:
        metrics_list_dict = {'D_top':[], 'fdr':[], 'tpr':[], 'fpr':[], 'shd':[], 'sid':[], 'nnz':[], 'precision':[], 'recall':[], 'F1':[], 'gscore':[]}
  
    metrics_res_dict = {}
    t_ls = []
    D_top = []
    orders = []
    simulation_seeds = [1, 2, 3, 4, 5]

    for i in range(args.runs):
        print(red('runs {}:'.format(i)))
        GT_DAG, train_set, _ = get_dataset(args.dataset, args.num_nodes, i, args.num_samples)
        data_ls = None
        
        if args.manualSeed:
            Seed = args.random_seed
        else:
            Seed = random.randint(1, 10000)
        print('Random Seed:', Seed)
        random.seed(Seed)
        torch.manual_seed(Seed)
        np.random.seed(Seed)
        if args.model in ['CaPS']:
            print(train_set)
            metrics_dict, order = train_test(args, train_set, GT_DAG, data_ls, runs=i)
            orders.append(order)
        else:
            raise Exception('No such model: {}'.format(args.model))

        error = order_divergence(order, GT_DAG)
        metrics_list_dict['D_top'].append(error)

        if only_order:
            pass
        else:
            for k in metrics_list_dict:
                if k=="D_top":
                    continue
                if np.isnan(metrics_dict[k]):
                    metrics_list_dict[k].append(0.0)
                else:
                    metrics_list_dict[k].append(metrics_dict[k])

    for k in metrics_list_dict:
        metrics_list_dict[k] = np.array(metrics_list_dict[k])
        metrics_res_dict[k] = '{}±{}'.format(np.around(np.mean(metrics_list_dict[k]), 5), np.around(np.std(metrics_list_dict[k]), 5))
        print(blue(str(k) + ': ' + metrics_res_dict[k]))

    orders_clean = [list(map(int, o)) for o in orders]
    order_str = json.dumps(orders_clean, separators=(',', ':')) 

    results_to_save = {
        "dataset": args.dataset,
        "n_nodes": GT_DAG.shape[0],
        "num_samples": args.num_samples,
        "order": order_str,
        "error": metrics_res_dict
    }

    json_path = "results_summary.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(results_to_save)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)