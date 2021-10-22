import torch.nn as nn

class Discriminator(nn.Module):
  '''
  Simple MLP classifier
  n_input: size of the inputs (observation dimension)
  n_hiddens: list of hidden layers sizes
  '''
  def __init__(self, n_input, n_hiddens, n_skills):
    super().__init__()
    layers = []
    layers.append(nn.Linear(n_input, n_hiddens[0]))
    layers.append(nn.ReLU())
    for i in range(len(n_hiddens)-1):
      layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
      layers.append( nn.ReLU() )
    
    self.head = nn.Sequential(*layers)
    self.output = nn.Linear(n_hiddens[-1], n_skills)
  

  def forward(self, x):
    return self.output(self.head(x))