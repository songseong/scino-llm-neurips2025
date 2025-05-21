import torch
import torch.nn as nn
import torch.fft
import numpy as np
import torch.nn.functional as F

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        # Initialize Wr as a learnable parameter
        self.Wr = nn.Parameter(torch.randn(F_dim // 2, M) * (gamma ** -1))
        
        # Define the MLP layers for processing Fourier features
        self.mlp = nn.Sequential(
            nn.Linear(F_dim, H_dim),
            nn.GELU(),
            nn.Linear(H_dim, D // G)
        )

    def forward(self, x):
        N, G, M = x.shape
        # Compute Fourier features (Eq. 2)
        projected = torch.matmul(x, self.Wr.T)
        F = (1 / np.sqrt(self.F_dim)) * torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
        
        # Apply MLP to Fourier features (Eq. 6)
        Y = self.mlp(F)
        
        # Reshape to match the expected output shape [N, D]
        PEx = Y.reshape((N, self.D))
        return PEx
        

class DiffFNO_LFPE(nn.Module):
    def __init__(self, n_nodes: int, n_fourier_layers: int = 1, bias: bool = False, norm_type: str = "batch") -> None:
        super().__init__()
        self.n_nodes = n_nodes

        self.n_fourier_layers = n_fourier_layers
        self.big_layer = max(1024, 5 * self.n_nodes) 
        self.small_layer = max(128, 3 * self.n_nodes)
        self.bias = bias
        self.norm_type = norm_type.lower()

        self.PositionalEncoding = LearnableFourierPositionalEncoding(G=1, M=self.n_nodes, F_dim=64, H_dim=32, D=self.big_layer, gamma=1.0)

        self.layer1 = nn.Sequential(
            nn.Linear(self.n_nodes, self.big_layer, self.bias),
            nn.LeakyReLU(),
            nn.LayerNorm(self.big_layer),
            nn.Dropout(0.2)
        )
        
        self.fourier_layers = nn.ModuleList()
        for _ in range(self.n_fourier_layers):
            norm_layer = self._get_norm_layer(2 * self.big_layer)
            self.fourier_layers.append(
                nn.Sequential(
                    nn.Linear(2 * self.big_layer, 2 * self.big_layer), 
                    nn.LeakyReLU(),
                    norm_layer
                )
            )
        
        self.layer2 = nn.Sequential(
            nn.Linear(self.big_layer, self.big_layer),
            nn.LeakyReLU(),
            nn.Linear(self.big_layer, self.small_layer),
            nn.LeakyReLU(),
            nn.Linear(self.small_layer, self.n_nodes)
        )

    def _get_norm_layer(self, dim):
        if self.norm_type == "batch":
            return nn.BatchNorm1d(dim)
        elif self.norm_type == "layer":
            return nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Choose 'layer' or 'batch'.")

    def forward(self, x, t):
        t = t.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_nodes)
        temb = self.PositionalEncoding(t)
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


class DiffFNO_LFPE_probe(nn.Module):
    def __init__(self, pretrained_fno, final_nodes):
        super().__init__()

        self.fno = pretrained_fno
        self._freeze_fno()

        self.n_nodes = self.fno.n_nodes
        self.big_layer = max(1024, 5 * self.n_nodes)
        self.small_layer = max(128, 3 * self.n_nodes)
        self.final_nodes = final_nodes

        self.linear_probe = nn.Sequential(
            nn.Linear(self.big_layer, self.big_layer),
            nn.LeakyReLU(),
            nn.Linear(self.big_layer, self.small_layer),
            nn.LeakyReLU(),
            nn.Linear(self.small_layer, self.n_nodes)
        )

    def _freeze_fno(self):
        for name in ['layer1', 'fourier_layers', 'PositionalEncoding']:
            module = getattr(self.fno, name)
            for param in module.parameters():
                param.requires_grad = False

    def lp_parameters(self):
        return self.linear_probe.parameters()
    
    def forward(self, x, t):
        t = t.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_nodes)
        temb = self.fno.PositionalEncoding(t)
        X = self.fno.layer1(x)
        for layer in self.fno.fourier_layers:
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
