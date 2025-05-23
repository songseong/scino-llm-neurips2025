import torch
import torch.nn as nn
import torch.fft
import numpy as np
import torch.nn.functional as F

class LearnableTimeEncoding(nn.Module):
    def __init__(self, F: int, M: int, H: int, D: int, gamma: float):
        super().__init__()
        self.F = F
        self.M = M
        self.H = H
        self.D = D
        self.gamma = gamma
        self.Wr = nn.Parameter(torch.randn(F, D) * (gamma ** -1))
        self.mlp = nn.Sequential(
            nn.Linear(2*F, M),
            nn.GELU(),
            nn.Linear(M, H)
        )

    def forward(self, t):
        projected = torch.matmul(t, self.Wr.T)
        fourier_feature = (1 / np.sqrt(2*self.F)) * torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
        LTE = self.mlp(fourier_feature)
        return LTE
        

class SciNO(nn.Module):
    def __init__(self, n_nodes: int, n_fourier_layers: int = 1, bias: bool = False, norm_type: str = "batch") -> None:
        super().__init__()
        self.n_nodes = n_nodes

        self.n_fourier_layers = n_fourier_layers
        self.H = max(1024, 5 * self.n_nodes) 
        self.S = max(128, 3 * self.n_nodes)
        self.bias = bias
        self.norm_type = norm_type.lower()

        self.TimeEncoding = LearnableTimeEncoding(F=32, M=32, H=self.H, D=self.n_nodes, gamma=5.0)

        self.layer1 = nn.Sequential(
            nn.Linear(self.n_nodes, self.H, self.bias),
            nn.LeakyReLU(),
            nn.LayerNorm(self.H),
            nn.Dropout(0.2)
        )
        
        self.fourier_layers = nn.ModuleList()
        for _ in range(self.n_fourier_layers):
            norm_layer = self._get_norm_layer(2 * self.H)
            self.fourier_layers.append(
                nn.Sequential(
                    nn.Linear(2 * self.H, 2 * self.H), 
                    nn.LeakyReLU(),
                    norm_layer
                )
            )
        
        self.layer2 = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.LeakyReLU(),
            nn.Linear(self.H, self.S),
            nn.LeakyReLU(),
            nn.Linear(self.S, self.n_nodes)
        )

    def _get_norm_layer(self, dim):
        if self.norm_type == "batch":
            return nn.BatchNorm1d(dim)
        elif self.norm_type == "layer":
            return nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Choose 'layer' or 'batch'.")

    def forward(self, x, t):
        t = t.unsqueeze(1).expand(-1, self.n_nodes)
        temb = self.TimeEncoding(t)
        X = self.layer1(x)

        for i, layer in enumerate(self.fourier_layers):
            X_skip = X
            X_fft = torch.fft.fftn(X, s=(X.shape[-1],), norm='ortho', dim=1) 
            X_fft = X_fft * temb 
            X_fft_real = X_fft.real.flatten(start_dim=1) 
            X_fft_imag = X_fft.imag.flatten(start_dim=1) 
            X_fft_combined = torch.cat([X_fft_real, X_fft_imag], dim=1) 
            X_fft_combined[:, 0::2] = X_fft_real
            X_fft_combined[:, 1::2] = X_fft_imag
            X_t = layer(X_fft_combined)
            X_t_real, X_t_imag = X_t[:,::2], X_t[:,1::2] 
            X_t_complex = torch.complex(X_t_real, X_t_imag) 
            X_ifft = torch.fft.ifftn(X_t_complex, dim=-1, norm='ortho').real
            X = X_ifft + X_skip

        X_t = self.layer2(X)
        
        return X_t


class SciNO_probe(nn.Module):
    def __init__(self, pretrained_scino, final_nodes):
        super().__init__()

        self.scino = pretrained_scino
        self._freeze_scino()

        self.n_nodes = self.scino.n_nodes
        self.H = max(1024, 5 * self.n_nodes)
        self.S = max(128, 3 * self.n_nodes)
        self.final_nodes = final_nodes

        self.linear_probe = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.LeakyReLU(),
            nn.Linear(self.H, self.S),
            nn.LeakyReLU(),
            nn.Linear(self.S, self.n_nodes)
        )

    def _freeze_scino(self):
        for name in ['layer1', 'fourier_layers', 'TimeEncoding']:
            module = getattr(self.scino, name)
            for param in module.parameters():
                param.requires_grad = False

    def lp_parameters(self):
        return self.linear_probe.parameters()
    
    def forward(self, x, t):
        t = t.unsqueeze(1).expand(-1, self.n_nodes)
        temb = self.scino.TimeEncoding(t)
        X = self.scino.layer1(x)
        for layer in self.scino.fourier_layers:
            X_skip = X
            X_fft = torch.fft.fftn(X, s=(X.shape[-1],), norm='ortho', dim=1) 
            X_fft = X_fft * temb 
            X_fft_real = X_fft.real.flatten(start_dim=1) 
            X_fft_imag = X_fft.imag.flatten(start_dim=1)
            X_fft_combined = torch.cat([X_fft_real, X_fft_imag], dim=1)
            X_fft_combined[:, 0::2] = X_fft_real
            X_fft_combined[:, 1::2] = X_fft_imag
            X_t = layer(X_fft_combined) 
            X_t_real, X_t_imag = X_t[:,::2], X_t[:,1::2]
            X_t_complex = torch.complex(X_t_real, X_t_imag) 
            X_ifft = torch.fft.ifftn(X_t_complex, dim=-1, norm='ortho').real
            X = X_ifft + X_skip
        return self.linear_probe(X)[:,self.final_nodes]
