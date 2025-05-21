import os
import tempfile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pygam import LinearGAM
from pygam.terms import TermList, SplineTerm
from cdt.utils.R import launch_R_script

from src.utils import np_to_csv

def train_gam(X, y, num_basis_functions):
    n, d = X.shape
    if n / d < 3 * num_basis_functions:
        num_basis_functions = int(np.ceil(n/(3*d)))
        print(f"Changed number of basis functions to {num_basis_functions} in order to have enough samples per basis function")
    terms = TermList()
    for i in range(d):
        terms += SplineTerm(i, n_splines=num_basis_functions)
    try: 
        mod_gam = LinearGAM(terms).gridsearch(X,y)
    except Exception as e:
        print(f"Warning: Error during GAM gridsearch: {e}")
        print("Fitting with no smoothing (lam=0).")
        terms = TermList()
        for i in range(d):
            terms += SplineTerm(i, n_splines=num_basis_functions, lam=0)
        mod_gam = LinearGAM(terms).fit(X,y)
    return np.array(mod_gam.statistics_['p_values'])


def select_gam(X, y, k, cutoff, num_basis_functions, verbose=False):
    d = X.shape[1]
    if d == 0:
        return np.array([])
    p_values = train_gam(X, y, num_basis_functions=num_basis_functions)
    if verbose:
        print(f"P-values: {p_values}")
    if p_values.shape[0] - 1 != d: 
        print("Warning: Unexpected number of p-values in select_gam.")
    selected_parents = p_values[:k] < cutoff
    return selected_parents

def pruning_python(G, X, cutoff=0.05, num_basis_functions=10, verbose=False, n_jobs = -1):
    d = G.shape[0] 
    final_graph = np.zeros((d, d))
    def prune(i):
        parents = np.where(G[:, i] == 1)[0] 
        if len(parents) > 0:
            selected_parents = select_gam(X[:, parents], X[:, i].reshape(-1,1), k=len(parents), cutoff=cutoff, num_basis_functions=num_basis_functions, verbose=verbose)
            final_parents = parents[selected_parents] 
        else:
            final_parents = np.array([], dtype=int)
        return final_parents
    results = Parallel(n_jobs=n_jobs)(
        delayed(prune)(i) for i in range(d)
        )
    for i, selected_parents in enumerate(results):
        final_graph[selected_parents, i] = 1
    return final_graph

def pruning_r(G, X, cutoff):
    with tempfile.TemporaryDirectory() as save_path:
        pruning_path = "diffan/pruning_R_files/cam_pruning.R"

        data_np = X 
        data_csv_path = np_to_csv(data_np, save_path)
        dag_csv_path = np_to_csv(G, save_path)

        arguments = dict()
        arguments['{PATH_DATA}'] = data_csv_path
        arguments['{PATH_DAG}'] = dag_csv_path
        arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
        arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
        arguments['{CUTOFF}'] = str(cutoff)
        arguments['{VERBOSE}'] = "FALSE" 

        def retrieve_result():
            G = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return G
        
        dag = launch_R_script(str(pruning_path), arguments, output_function=retrieve_result)
    return dag

def pruning(G, X, cutoff, method="python"):
    if method.lower() == "python":
        return pruning_python(G, X, cutoff)
    elif method.lower() == "r":
        return pruning_r(G, X, cutoff)
    else:
        raise ValueError(f"Unknown pruning method: {method}. Use 'python' or 'r'.")