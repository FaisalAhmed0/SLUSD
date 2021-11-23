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
    for i in range(len(n_hiddens)-1):
      layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
      layers.append( nn.ReLU() )
      # TODO: Added dropout
      if dropout:
        layers.append( nn.Dropout(dropout) )
    
    self.head = nn.Sequential(*layers)
    self.output = nn.Linear(n_hiddens[-1], n_skills)
    # TODO: print head to check
    print(f"Discriminator head {self.head}")
  

  def forward(self, x):
    return self.output(self.head(x))