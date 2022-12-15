import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import utils


class EncNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(EncNet, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.bn_0 = nn.BatchNorm1d(input_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(2)])
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_sigma = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x) -> tuple():
        x = self.bn_0(x)
        out = F.relu(self.bn_1(self.linear_in(x)))
        
        for l, bn in zip(self.linears, self.bns):
            out = F.relu(bn(l(out)))
        
        mu = self.linear_mu(out)
        log_var = self.linear_sigma(out)
        
        return mu, log_var

class EncGaussian(nn.Module):
    def __init__(self, enc):
        super(EncGaussian, self).__init__()
        self.enc = enc
    
    def forward(self, x) -> tuple():
        mean, log_var = self.enc(x)
        var = torch.exp(log_var)
        z = utils.sample_normal(mean, var)
        return z, mean, var

class DecNet(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super(DecNet, self).__init__()
        self.linear_in = nn.Linear(latent_dim, hidden_dim)
        self.site_in = nn.Linear(4, hidden_dim)
        self.bn_0 = nn.BatchNorm1d(latent_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z, site_assignments=None):
        z = self.bn_0(z)
        out = self.linear_in(z)
        if site_assignments is not None:
            out += self.site_in(site_assignments)
        out = F.relu(self.bn_1(out))
        
        for l, bn in zip(self.linears, self.bns):
            out = F.relu(bn(l(out)))
        
        out = self.linear_out(out)
        return out

class DecGaussian(nn.Module):
    def __init__(self, dec):
        super(DecGaussian, self).__init__()
        self.dec = dec
        self.log_scale = nn.Parameter(torch.Tensor([0.]))
    
    def forward(self, z, x, site_assignments=None):
        out = self.dec(z, site_assignments)
        recon_loss = utils.log_prob_normal(out, torch.exp(self.log_scale), x)
        return out, recon_loss

class VAE(nn.Module):
    def __init__(self, enc, dec):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x, site_assignments=None):
        z, mu, sigma = self.enc(x)
        decoder_out, recon_loss = self.dec(z, x, site_assignments)
        Dkl = utils.kl_divergence(mu, sigma)
        elbo = Dkl - recon_loss
        if site_assignments is not None:
            z = torch.cat([z, site_assignments], dim=1)
        return elbo, Dkl, recon_loss, z, decoder_out

class DecNegBinom(nn.Module):
    def __init__(self, dec):
        super(DecNegBinom, self).__init__()
        self.dec = dec
        self.event_prob = nn.Parameter(torch.Tensor([0.]))

    def forward(self, z, x, site_assignments=None):
        mean = 1.0001 + F.elu(self.dec(z, site_assignments))
        event_prob_ = torch.sigmoid(self.event_prob) 
        n_succ = mean * event_prob_ / (1 - event_prob_)
        recon_loss = utils.log_prob_neg_binom(n_succ, event_prob_, x)
        return mean, recon_loss