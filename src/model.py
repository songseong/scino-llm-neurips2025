import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import wandb 
from torch.optim.lr_scheduler import ReduceLROnPlateau


from src.gaussian_diffusion import GaussianDiffusion, UniformSampler, get_named_beta_schedule, mean_flat, \
                                         LossType, ModelMeanType, ModelVarType
from src.nn import DiffMLP
from src.scino import SciNO, SciNO_probe

from torch.utils.data import TensorDataset, DataLoader
from src.ordering import topological_ordering
from datetime import datetime
import time
from src.pruning import pruning
from src.sde import VPSDE
from src.utils import full_DAG, compute_mmd, num_errors, namespace2dict
from sklearn.model_selection import train_test_split
import json


def score_loss(model, sde, x0, t):
    x_t = sde.forward(x0, t)
    score = x_t.noise
    output = model(x_t, t.float())
    weight = output - score
    loss = (weight).square().sum(dim=(1)).mean(dim=0)

    return loss

class Model:
    def __init__(self, config, n_nodes, model_type):
        self.config = config
        self.n_nodes = n_nodes
        self.model_type = model_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model_type == "scino":
            self.sde = VPSDE(config)
        self.n_steps = 100

        self.model, self.epochs, self.learning_rate, self.batch_size, self.early_stopping_wait = self.initialize_model(config, n_nodes, model_type)
        self.ordering_batch_size = getattr(config.evaluation, "ordering_batch_size", self.batch_size)
        self.ordering_option = getattr(config.evaluation, "ordering_option", None)
        self.best_model = getattr(config.evaluation, "best_model", False)
        self.grad_clip = getattr(config.training, "grad_clip", None)
        self.mmd_model = getattr(config.training, "mmd_model", False)

        self.model.float().to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay=0.01)
        self.best_loss, self.best_mmd = float("inf"), float("inf")

        self.scheduler_type = getattr(self.config.training, 'scheduler', 'plateau').lower()

        if self.scheduler_type == "onecycle":
            steps_per_epoch = self.config.datasets.num_samples // self.batch_size  
            total_steps = steps_per_epoch * self.epochs
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.opt,
                max_lr=1e-3,
                total_steps=total_steps,
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1e4,
                anneal_strategy='cos',
                verbose=True
            )
        elif self.scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        # Pruning
        self.cutoff = config.evaluation.cutoff
        self.prun_p = config.evaluation.pruning_P

        self.file_name = self.get_file_name(config, model_type)
        self.should_log = (self.config.training.graph_idx == 0)

    def initialize_model(self, config, n_nodes, model_type):
        if model_type == "scino":
            n_fourier_layers = getattr(config.model, "n_fourier_layers", None)
            bias = getattr(config.model, "bias", False)
            norm_type = getattr(config.model, "norm_type", None)
            model = SciNO(n_nodes, n_fourier_layers=n_fourier_layers, bias=bias, norm_type=norm_type)
            epochs = config.training.epochs
            learning_rate = config.training.learning_rate
            batch_size = config.training.batch_size
            early_stopping_wait = config.training.early_stopping_wait
        else:
            betas = get_named_beta_schedule("linear", self.n_steps, scale=1, beta_start=0.0001, beta_end=0.02)
            self.gaussian_diffusion = GaussianDiffusion(betas=betas, loss_type=LossType.MSE, model_mean_type=ModelMeanType.EPSILON, model_var_type=ModelVarType.FIXED_LARGE, rescale_timesteps=True)
            self.schedule_sampler = UniformSampler(self.gaussian_diffusion)
            model = DiffMLP(n_nodes)
            epochs = config.training.epochs 
            learning_rate = config.training.learning_rate
            batch_size = config.training.batch_size
            early_stopping_wait = config.training.early_stopping_wait
        return model, epochs, learning_rate, batch_size, early_stopping_wait


    def get_file_name(self, config, model_type):
        
        if config.dataset_name == "synthetic":
            graph_idx = config.training.graph_idx
            n_node = config.training.n_node
            model_file_name = f'{model_type}_n{n_node}_g{graph_idx}.pth'
            model_file_path = os.path.join("./ckpt/", model_file_name)
        else:
            graph_idx= config.training.graph_idx
            model_file_name = f'{model_type}_{config.dataset_name}_g{graph_idx}.pth'
            if config.pretrain:
                model_file_path = os.path.join(f"./ckpt/ckpt_{config.dataset_name}/", model_file_name)
            else:
                model_file_path = os.path.join(f"./ckpt/", model_file_name)
        return model_file_path


    def fit(self, X, true_causal_matrix):
        X = X.to(self.device)
        order_name = f"order_{self.ordering_option}" if self.ordering_option else "order"

        if self.config.load_ckpt:
            ckpt = torch.load(self.file_name, weights_only=True)
            print(ckpt.keys())
            if self.best_model:
                self.model.load_state_dict(ckpt["best_model_state_dict"])
            elif "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            if self.config.evaluation.load_order == True:
                self.order = ckpt.get(order_name, None)

            else: 
                self.order = topological_ordering2(self.model, X, self.n_nodes, self.device, self.ordering_batch_size, self.config)
                    
                ckpt[order_name] = self.order
                torch.save(ckpt, self.file_name)
        else:
            self.train_score(X)
            self.model_state = self.model.state_dict()
            if self.best_model:
                self.model.load_state_dict(self.best_model_state)
            
            self.order = topological_ordering(self.model, X, self.n_nodes, self.device, self.ordering_batch_size, self.config)

            
            if wandb.run:
                run_dir = f"./ckpt_Diff/{self.config.setup.experiment_name}/{wandb.run.id}/"
            else:
                import datetime
                run_dir = f"./ckpt_Diff/{self.config.setup.experiment_name}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
            os.makedirs(run_dir, exist_ok=True)
            self.file_name = os.path.join(run_dir, os.path.basename(self.file_name))

            torch.save({"model_state_dict": self.model_state,
                        "epoch": self.epochs,
                        "optimizer": self.opt.state_dict(),
                        "best_model_state_dict": self.best_model_state,
                        "best_epoch": self.best_model_state_epoch,
                        order_name: self.order}, self.file_name)
            

        out_dag = None
        if self.prun_p:
            out_dag = pruning(full_DAG(self.order), X.detach().cpu().numpy(), self.cutoff)
        
        return out_dag, self.order 

    def compute_loss(self, x_0):
        x_0 = x_0.float().to(self.device)

        if self.model_type == "scino":
            t = torch.rand(x_0.shape[0]).to(self.device) * self.config.training.T
            return score_loss(self.model, self.sde, x_0, t)
        else:
            t, _ = self.schedule_sampler.sample(x_0.shape[0], self.device)
            noise = torch.randn_like(x_0).to(self.device)
            x_t = self.gaussian_diffusion.q_sample(x_0, t, noise=noise)
            model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t))
            return (noise - model_output).square().sum(dim=(1)).mean()

    def train_score(self, X):
        self.model.train()
        self.best_model_state_epoch = self.early_stopping_wait
        self.best_model_state = None

        n_samples = X.shape[0]
        val_ratio = 0.2
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - val_size
        X_train, X_val = X[:train_size], X[train_size:]
        data_loader_val = DataLoader(X_val, min(val_size, self.batch_size))
        data_loader = DataLoader(X_train, min(train_size, self.batch_size), drop_last=True)

        pbar = tqdm(range(self.epochs), desc="Training Epoch")
        for epoch in pbar:
            loss_per_step = []

            for x_0 in data_loader:
                diffusion_loss = self.compute_loss(x_0)
                loss_per_step.append(diffusion_loss.item())
                self.opt.zero_grad()
                diffusion_loss.backward()
                if self.grad_clip is not None and epoch >= 100:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip) 
                self.opt.step()
                if self.scheduler_type == "onecycle":
                    self.scheduler.step()

                self.log_wandb("Training loss", diffusion_loss)

            if epoch % 10 == 0:
                with torch.no_grad():
                    loss_per_step_val = []
                    for x_0 in data_loader_val:
                        diffusion_loss = self.compute_loss(x_0)
                        loss_per_step_val.append(diffusion_loss.item())
                    epoch_val_loss = np.mean(loss_per_step_val)
                    
                    if (self.config.dataset_name != "synthetic") and (self.scheduler_type != "onecycle"):
                        self.scheduler.step(epoch_val_loss)

                self.log_wandb("Validation loss", epoch_val_loss)
                pbar.set_postfix({'Epoch Loss': epoch_val_loss})

                if self.mmd_model:
                    with torch.no_grad():
                        x_init = torch.randn(1000, self.n_nodes).to(self.device)
                        samples = self.sde.sample(x_init, num_steps=1000, model=self.model)
                        mmd_score = compute_mmd(X_train[0:1000].detach().cpu().numpy(), samples.detach().cpu().numpy())
                        print(f"Epoch {epoch+1}/{self.epochs}, MMD={mmd_score:.4f}")

                        if self.best_mmd > mmd_score:
                            self.best_mmd = mmd_score
                            self.best_model_state = copy.deepcopy(self.model.state_dict())
                            self.best_model_state_epoch = epoch
                            if self.config.use_wandb:
                                wandb.run.summary[f"best_mmd"] = mmd_score

                else:
                    if self.best_loss > epoch_val_loss:
                        self.best_loss = epoch_val_loss
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.best_model_state_epoch = epoch
                

        if self.config.training.model_save:
            torch.save({"model_state_dict": self.model.state_dict(),
                        "epoch": self.epochs,
                        "optimizer": self.opt.state_dict(),
                        "best_model_state_dict": self.best_model_state,
                        "best_epoch": self.best_model_state_epoch}, self.file_name)
            print(f"Model saved at epoch {self.best_model_state_epoch + 1}, {self.file_name}")


    def log_wandb(self, loss_key, loss_value):
        if self.should_log and self.config.use_wandb:
            key = f'{"DiffAN" if self.model_type == "mlp" else "scino"} - {loss_key}'
            wandb.log({key: loss_value})


class ModelProbe(Model):
    def __init__(self, config, n_nodes, model_type):
        super().__init__(config, n_nodes, model_type)
        self.probe_lr = self.config.training.probe_lr
        self.probe_epochs = getattr(self.config.training,"probe_epochs", 50)

    def probing(self, X, true_adj):
        if self.config.pretrain is None:
            ckpt = torch.load(self.file_name, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
        elif self.config.pretrain==True:
            self.train_score(X)
        order = self.topological_ordering(X)
        errors = num_errors(order, true_adj)

        # save results
        results = {"order": [int(v) for v in order], 
                   "num_errors": float(errors),
                   "config": namespace2dict(self.config)}
        
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"prob_results/{self.config.dataset_name}/{experiment_name}"
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, f"{self.model_type}_{self.config.dataset_name}.json")
        with open(file_path, "w") as f:
            json.dump(results, f)

        out_dag = None
        if self.prun_p:
            out_dag = pruning(full_DAG(self.order), X.detach().cpu().numpy(), self.cutoff)
        return out_dag, order


    def Stein_hess(self, X, stein_batch_size, eta_G=0.001, eta_H=0.001, s = None, device='cuda'):
        """
        Estimates the diagonal of the Hessian of log p_X at the provided samples points
        X, using first and second-order Stein identities
        """
        n, d = X.shape
        X_diff = X.unsqueeze(1)-X
        if s is None:
            D = torch.norm(X_diff, dim=2, p=2)
            s = D.flatten().median()
        K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
        
        nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
        G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n).to(device)), nablaK)
        if stein_batch_size:
            nabla2K_parts = []

            for i in range(0, n, stein_batch_size):
                X_batch = X[i:i+stein_batch_size]
                X_diff_batch = X_batch.unsqueeze(1) - X  # shape (B, n, d)
                K_batch = torch.exp(-torch.norm(X_diff_batch, dim=2)**2 / (2 * s**2)) / s
                part = torch.einsum('kij,ki->kj', -1/s**2 + X_diff_batch**2/s**4, K_batch)
                nabla2K_parts.append(part)
            nabla2K = torch.cat(nabla2K_parts, dim=0)
            
        else: 
            nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
        output = (-G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n).to(device)), nabla2K)).to('cpu')
        return output
    
    def post_train(self, model, X, active_nodes):
        model.to(self.device)
        model.train()
        batch_size = 128 

        X_train, X_valid = train_test_split(X, test_size=0.1, random_state=42)

        train_loader = DataLoader(X_train, batch_size, drop_last=True, shuffle=True)
        valid_loader = DataLoader(X_valid, batch_size, drop_last=False, shuffle=False)

        def loss_fn(x, stein_hessian):
            x = x.float().to(self.device)
            t = torch.tensor([0.0], device=self.device)
            output = model(x,t)
            return (output - stein_hessian).pow(2).mean()
        
        stein_batch_size = 300 if self.config.dataset_name.startswith("arth150") else 1000
        stein_batch_size = min(128, X.shape[0])

        pbar = tqdm(range(self.probe_epochs), desc="Training Epoch")
        
        for epoch in pbar:
            loss_per_step = []
            for x_batch in train_loader:
                with torch.no_grad():
                    y_stein = self.Stein_hess(x_batch[:,active_nodes], stein_batch_size=stein_batch_size).to(self.device)
                loss = loss_fn(x_batch, y_stein)
                loss_per_step.append(loss.item())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.log_wandb("Training loss", loss)  
            if epoch % 10 == 0:
                with torch.no_grad():
                    loss_per_step_val = []
                    for x_batch in valid_loader:
                        with torch.no_grad():
                            y_stein = self.Stein_hess(x_batch[:,active_nodes], stein_batch_size=stein_batch_size).to(self.device)
                        loss = loss_fn(x_batch, y_stein)
                        loss_per_step_val.append(loss.item())
                epoch_val_loss = np.mean(loss_per_step_val)
            self.log_wandb("Validation loss", epoch_val_loss)
            pbar.set_postfix({'Epoch Loss': epoch_val_loss})
        return
    
    def topological_ordering(self, X):
        order = []
        n_nodes = X.shape[1]
        active_nodes = list(range(n_nodes))

        pbar = tqdm(range(n_nodes - 1), desc="Nodes ordered")
        for _ in pbar:
            probe_model = SciNO_probe(self.model, active_nodes)
            self.opt = torch.optim.Adam(probe_model.lp_parameters(), lr=self.probe_lr)
            self.post_train(probe_model, X, active_nodes)
            t = torch.tensor([0.0], device=self.device)
            outputs = []
            batch_size = 128
            with torch.no_grad():
                for i in range(0, X.shape[0], batch_size):
                    x_batch = X[i:i+batch_size]
                    out_batch = probe_model(x_batch, t)  
                    outputs.append(out_batch)
                output = torch.cat(outputs, dim=0)  
                if self.config.CaPS:
                    tqdm.write("CaPS ordering")
                    leaf_ = int(output.mean(0).argmax())
                else:
                    tqdm.write("DiffAN ordering")
                    leaf_ = int(output.var(0).argmin())
            leaf_global = active_nodes[leaf_]
            order.append(leaf_global)
            active_nodes.pop(leaf_)
        order.append(active_nodes[0])
        order.reverse()
        return order