import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
  '''
  Simple MLP classifier
  n_input: size of the inputs (observation dimension).
  n_hiddens: list of hidden layers sizes.
  n_skills: Number of skills.
  dropout: dropout probability.
  '''
  def __init__(self, n_input, n_hiddens, n_skills, dropout=None):
    super().__init__()
    layers = []
    # layers.append(nn.Linear(n_input, n_skills))
    layers.append(nn.Linear(n_input, n_hiddens[0]))
    layers.append(nn.ReLU())
    if dropout:
          layers.append( nn.Dropout(dropout) )
    for i in range(len(n_hiddens)-1):
      layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
      layers.append( nn.ReLU() )
      # TODO: Added dropout
      if dropout:
        layers.append( nn.Dropout(dropout) )
    
    self.head = nn.Sequential(*layers)
    print(self.head)
    self.output = nn.Linear(n_hiddens[-1], n_skills)

  def forward(self, x):
    # return self.head(x)
    return self.output(self.head(x))

# An Encoder network for different MI critics
class Encoder(nn.Module):
    def __init__(self,  n_input, n_hiddens, n_latent, dropout=None):
        super().__init__()
        layers = []
        # layers.append(nn.Linear(n_input, n_skills))
        layers.append(nn.Linear(n_input, n_hiddens[0]))
        layers.append(nn.ReLU())
        if dropout:
              layers.append( nn.Dropout(dropout) )
        for i in range(len(n_hiddens)-1):
          layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
          layers.append( nn.ReLU() )
          # TODO: Added dropout
          if dropout:
            layers.append( nn.Dropout(dropout) )

        layers.append(nn.Linear(n_hiddens[-1], n_latent))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# Separable Critic
class SeparableCritic(nn.Module):
    def __init__(self, state_dim, skill_dim, hidden_dims, latent_dim, temperature=1):
        super().__init__()
        self.temp = temperature
        self.num_skills = skill_dim
        # State encoder
        self.state_enc = Encoder(state_dim, hidden_dims, latent_dim)
        # Skill encoder
        self.skill_enc = Encoder(skill_dim, hidden_dims, latent_dim)
    def forward(self, x, y):
        x = F.normalize(self.state_enc(x), dim=-1) # shape (B * latent)
        y = F.normalize(self.state_enc(y), dim=-1) # shape (B * latent)
        return torch.sum(x[:, None, :] * y[None, :, :], dim=-1) / self.temp #shape (B * B)
    
# Concatenate Critic
class ConcatCritic(nn.Module):
    def __init__(self, state_dim, skill_dim, hidden_dims, temperature=1):
        super().__init__()
        self.temp = temperature
        self.num_skills = skill_dim
        self.mlp = Encoder(state_dim+skill_dim, hidden_dims, 1)
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
        covar = torch.exp(log_var) + 1e-3
        self.actions_dist = MultivariateNormal(mean.squeeze(dim=0), torch.diag(covar.squeeze(dim=0)))
        action = self.actions_dist.sample()
        log = self.actions_dist.log_prob(action)
        return action

    def entropy(self):
        return self.actions_dist.entropy()


    



