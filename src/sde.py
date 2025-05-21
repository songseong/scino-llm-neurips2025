import torch
import copy

class VPSDE:
    def __init__(self, config, threshold=None):
        self.beta_0 = config.model.beta_0
        self.beta_1 = config.model.beta_1
        self.T = config.model.T
        self.threshold = threshold
    
    def beta(self, t):
        return (self.beta_1 - self.beta_0) * t + self.beta_0

    def marginal_log_mean_coeff(self, t):
        return -1 / 4 * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / 2 * t * self.beta_0

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.pow(1. - torch.exp(self.marginal_log_mean_coeff(t) * 2), 1 / 2)
    
    def f(self, t):
        return -self.beta(t)/2
    
    def g(self, t):
        return torch.sqrt(self.beta(t))
    
    def square_g(self, t):
        return self.beta(t)
    
    def forward(self, x, t):
        x_t_mean = self.diffusion_coeff(t)[:, None] * x
        x_t_std = self.marginal_std(t)
        noise = torch.randn_like(x).to(x.device)
        x_t = x_t_mean + x_t_std[:, None] * noise
        x_t.noise = noise
        return x_t
        
    def norm_grad(self, x):
        size = x.shape
        l = len(x)

        x = x.reshape((l, -1))
        indices = x.norm(dim=1) > self.threshold
        x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * self.threshold

        x = x.reshape(size)
        return x

    def sample_step(self, x, s, t, model):
        """
        single step of the reverse SDE of diffusion model
        """
        delta_t = t - s
        
        score_pred = - model(x, t) / self.marginal_std(t)[:, None]
        
        drift_term = (self.f(t)[:, None] * x - self.square_g(t)[:, None] * score_pred) * delta_t[:,None]
        diffusion_term = self.g(t)[:, None] * torch.randn_like(x).to(x.device) * torch.sqrt(-delta_t)[:, None]
        
        if self.threshold:
            x = self.norm_grad(x)

        return x + drift_term + diffusion_term

    @torch.no_grad()
    def sample(self, x_init, num_steps, model, eps=1e-5):
        """
        sample from the diffusion model
        """
        self.device = "cuda"
        x = copy.deepcopy(x_init).to(self.device)
        batch_size = x.shape[0]
        timesteps = torch.linspace(self.T, eps, num_steps+1, device=self.device)

        for i in range(num_steps):
            vec_s = torch.ones((batch_size, ), device=self.device) * timesteps[i]
            vec_t = torch.ones((batch_size, ), device=self.device) * timesteps[i+1]
            x = self.sample_step(x, vec_s, vec_t, model)

        return x