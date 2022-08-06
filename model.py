from datetime import datetime
import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, num_inputs, num_hidden, num_outputs):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(num_inputs, num_hidden),
      nn.ReLU(),
      nn.Linear(num_hidden, num_outputs)
    )

  def forward(self, x):
    return self.net(x)

  def save(self, path=None):
    t = datetime.now()
    if path is None:
      path = f'wnb-{t.year}-{t.month:0>2}-{t.day:0>2}-{t.hour:0>2}-{t.minute:0>2}'
    torch.save(self.state_dict(), path)

  def load(self, path):
    self.load_state_dict(torch.load(path))
    self.eval()
