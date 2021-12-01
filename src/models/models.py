import torch
import torch.nn as nn

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
    return self.output(self.head(x))


'''
The classifier parametrization
	s_enc = Sequential([Dense(64, ‘relu’), Dense(64, ‘relu’), Dense(32, ‘linear’)])
	z_enc = Sequential([Dense(64, ‘relu’), Dense(64, ‘relu’), Dense(32, ‘linear’)])
	s_repr = s_enc(s)  // [batch_size x 32]
	z_repr = z_enc(z)
	score = sum(s_repr * z_repr, axis=1)  // [batch_size]
	score_outer = sum(s_repr[:, None, :] * z_repr[None, :, :], axis=2)  // [batch_size x batch_size]
f(s, z) -> real number
p(z | s) = f(s, z) / \sum_{z’} f(s, z’)
'''

#TODO: Added the dot product based classifier
class Encoder(nn.Module):
    def __init__(self, state_n_input, skill_n_input, n_hiddens, n_latent, dropout=None):
      super().__init__()
      # define the state encoder
      self.num_skills = skill_n_input
      state_enc_layers = []
      state_enc_layers.append(nn.Linear(state_n_input, n_hiddens[0]))
      state_enc_layers.append(nn.ReLU())
      if dropout:
          state_n_input.append(nn.Dropout(dropout))
      for i in range(len(n_hiddens)-1):
        state_n_input.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
        state_n_input.append( nn.ReLU() )
        # TODO: Added dropout
        if dropout:
          state_n_input.append( nn.Dropout(dropout) )

      state_enc_layers.append(nn.Linear(n_hiddens[-1], n_latent))
      self.state_enc = nn.Sequential(*state_enc_layers)

      # define the skills encoder
      skill_enc_layers = []
      skill_enc_layers.append(nn.Linear(skill_n_input, n_hiddens[0]))
      skill_enc_layers.append(nn.ReLU())
      if dropout:
          skill_enc_layers.append(nn.Dropout(dropout))
      for i in range(len(n_hiddens)-1):
        skill_enc_layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
        skill_enc_layers.append( nn.ReLU() )
        # TODO: Added dropout
        if dropout:
          skill_enc_layers.append( nn.Dropout(dropout) )

      skill_enc_layers.append(nn.Linear(n_hiddens[-1], n_latent))
      self.skill_enc = nn.Sequential(*skill_enc_layers)

      
      
    def forward(self, state, skill):
        # pass the state to the state encoder
        state_rep = self.state_enc(state)
        # pass the skill to the skill encoder
        skill_rep = self.skill_enc(skill)
        # compute the dot product score
        score =  torch.sum(state_rep * skill_rep, dim=1)
        score = torch.exp( score - score.max() )
        # compute the outer product score
        score_all_skills = self.score_all(state_rep)
        # compute the final output
        output = score / score_all_skills
        print(f"output of the encoder: {output}")
        return output

    def softmax(self, score, score_outer):
        max_score = torch.max(score_outer)
        numerator = torch.exp( score - max_score )
        denominator = torch.sum( torch.exp(score_outer - max_score), dim=1)
        return numerator / denominator

    @torch.no_grad()
    def score_all(self, state_rep):
        batch_size = state_rep.shape[0]
        scores = torch.zeros(batch_size)
        for skill in range(self.num_skills):
            skill_onehot = torch.zeros(batch_size, self.num_skills)
            skill_onehot[torch.arange(batch_size), skill] = 1
            skill_rep = self.skill_enc(skill_onehot)
            score = torch.sum(state_rep * skill_rep, dim=1)
            score = torch.exp(score - score.max())
            print(f"Score in score all shape: {score.shape}")
            scores += score
        return scores

        
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


    



