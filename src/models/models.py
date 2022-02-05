import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import MultivariateNormal
from  torch.nn.utils.parametrizations import spectral_norm

class Discriminator(nn.Module):
    '''
    Simple MLP classifier
    n_input: size of the inputs (observation dimension).
    n_hiddens: list of hidden layers sizes.
    n_skills: Number of skills.
    dropout: dropout probability.
    '''
    def __init__(self, n_input, n_hiddens, n_skills, dropout=None, sn=None, bn=None):
        super().__init__()
        layers = []
        if sn:
            layers.append( spectral_norm(nn.Linear(n_input, n_hiddens[0])))
        else:
            layers.append(nn.Linear(n_input, n_hiddens[0]))
            
        layers.append(nn.ReLU())
        if dropout:
            layers.append( nn.Dropout(dropout) )
        for i in range(len(n_hiddens)-1):
            if sn:
                layers.append( spectral_norm(nn.Linear(n_hiddens[i], n_hiddens[i+1]) ))
            else:
                layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
            layers.append( nn.ReLU() )
            if dropout:
                layers.append( nn.Dropout(dropout) )
        self.layers = layers
        self.layers.append(nn.Linear(n_hiddens[-1], n_skills))
        self.model = nn.Sequential(*layers)
        # self.output = nn.Linear(n_hiddens[-1], n_skills)

    def forward(self, x):
        return self.model(x)
        # return self.output(self.head(x))

# An Encoder network for different MI critics
class Encoder(nn.Module):
    def __init__(self,  n_input, n_hiddens, n_latent, dropout=None):
        super().__init__()
        layers = []
        layers.append(nn.Linear(n_input, n_hiddens[0]))
        layers.append(nn.ReLU())
        if dropout:
              layers.append( nn.Dropout(dropout) )
        for i in range(len(n_hiddens)-1):
          layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
          layers.append( nn.ReLU() )
          if dropout:
            layers.append( nn.Dropout(dropout) )

        layers.append(nn.Linear(n_hiddens[-1], n_latent))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Separable Critic
# Based on https://github.com/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
class SeparableCritic(nn.Module):
    def __init__(self, state_dim, skill_dim, hidden_dims, latent_dim, temperature=1, dropout=None):
        super().__init__()
        self.temp = temperature
        self.num_skills = skill_dim
        # State encoder
        self.state_enc = Encoder(state_dim, hidden_dims, latent_dim, dropout=dropout)
        # Skill encoder
        self.skill_enc = Encoder(skill_dim, hidden_dims, latent_dim, dropout=dropout)
    def forward(self, x, y):
        x = F.normalize(self.state_enc(x), dim=-1) # shape (B * latent)
        y = F.normalize(self.skill_enc(y), dim=-1) # shape (B * latent)
        return torch.sum(x[:, None, :] * y[None, :, :], dim=-1) / self.temp #shape (B * B)
    
# Concatenate Critic
# Based on https://github.com/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
class ConcatCritic(nn.Module):
    def __init__(self, state_dim, skill_dim, hidden_dims, temperature=1, dropout=None):
        super().__init__()
        self.temp = temperature
        self.num_skills = skill_dim
        self.mlp = Encoder(state_dim+skill_dim, hidden_dims, 1, dropout=dropout)
    def forward(self, x, y):
        batch_size = x.shape[0]
        # tile x with the batch size
        x_tiled = torch.tile(x[None, :], (batch_size, 1, 1))
        # tile y with the batch size
        y_tiled = torch.tile(y[:, None], (1, batch_size, 1))
        # xy_pairs
        xy_pairs = torch.cat((x_tiled, y_tiled), dim=2).reshape(batch_size*batch_size, -1)
        # compute the scores
        scores = self.mlp(xy_pairs) / self.temp
        return scores.reshape(batch_size, batch_size).T
        
        
# Created for ES         
class MLP_policy(nn.Module):
    def __init__(self, n_input, n_hiddens, n_actions):
        super().__init__()
        layers = []
        layers.append(nn.Linear(n_input, n_hiddens[0]))
        layers.append(nn.ReLU())
        for i in range(len(n_hiddens)-1):
            layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
            layers.append( nn.ReLU() )

        self.model = nn.Sequential(*layers)
        self.mean = nn.Linear(n_hiddens[-1], n_actions)
        self.log_var = nn.Linear(n_hiddens[-1], n_actions)

    def forward(self, x):
        x = self.model(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var
  
    def sample_action(self, x):
        mean, log_var = self.forward(x)
        covar = torch.exp(torch.clip(log_var, -1, 5))
        # print(covar.shape[1] == 1)
        if covar.shape[1] == 1:
            self.actions_dist = MultivariateNormal(mean, covar)
        else:
            self.actions_dist = MultivariateNormal(mean.squeeze(dim=0), torch.diag(covar.squeeze(dim=0)))
        action = self.actions_dist.sample()
        log = self.actions_dist.log_prob(action)
        return action

    def entropy(self):
        return self.actions_dist.entropy()


    



