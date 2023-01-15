import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class FFMLayer(nn.Module):
  def __init__(self, rep_dim, L=10):
    super().__init__()
    self.coefs = (torch.arange(start=1, end=L+1e-12) * math.pi * 0.5).cuda()

  def forward(self, x):
    argument = torch.kron(self.coefs, x)
    return torch.hstack((torch.sin(argument), torch.cos(argument)))

class FourierNet(nn.Module):
  def __init__(self,
         in_features,
         hidden_features,
         hidden_blocks,
         out_features,
         L = 10):
    super().__init__()

    self.ffm = FFMLayer(in_features, L)
    ffm_expansion_size = 2*in_features*L

    self.blocks = []

    ### First block
    self.blocks.append(nn.ModuleList([
      nn.Linear(ffm_expansion_size, hidden_features),
      nn.Linear(hidden_features, hidden_features)
      ]))

    ### Hidden block
    for i in range(hidden_blocks-1):
      self.blocks.append(nn.ModuleList([
        nn.Linear(hidden_features + ffm_expansion_size, hidden_features),
        nn.Linear(hidden_features, hidden_features)
      ]))

    ### Final
    self.final_block = [
      nn.Linear(hidden_features + ffm_expansion_size, hidden_features),
      nn.Linear(hidden_features, int(hidden_features / 2)),
      nn.Linear(int(hidden_features / 2), out_features)
      ]

    self.blocks = nn.ModuleList(self.blocks)
    self.final_block = nn.ModuleList(self.final_block)

  def forward(self, coords):
    ffm_out = self.ffm(coords)
    x = ffm_out

    for b in range(len(self.blocks)):
      fcs = self.blocks[b]
      x = fcs[0](x)
      x = F.relu(x)
      x = fcs[1](x)
      x = F.relu(x)
      x = torch.cat((x, ffm_out), dim=1)
    
    x = self.final_block[0](x)
    x = F.relu(x)
    x = self.final_block[1](x)
    x = F.relu(x)
    x = self.final_block[2](x)

    return x

class FourierNetBlock(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    
    self.fc1 = nn.Linear(in_features, out_features)
    self.fc2 = nn.Linear(out_features, out_features)
    
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    return x
    
class FourierNetFinalBlock(nn.Module):
  def __init__(self, in_features, hidden_features, out_features):
    super().__init__()
    
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.fc2 = nn.Linear(hidden_features, int(hidden_features / 2))
    self.fc3 = nn.Linear(int(hidden_features / 2), out_features)
      
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    return x


def unit_test_ffm_dimension():
  import ops, utils

  operator = ops.ParallelBeamGeometryOp(64, 60, 500)
  grid_params = {
    'angles': torch.cos(torch.tensor(operator.angles, dtype=torch.float32)),
    'num_detectors': operator.num_detectors
    }
  grid = utils.get_sino_mgrid(**grid_params)
  print (grid.shape)

  ffm = FFMLayer(rep_dim = grid.shape[1], L = 10)
  ffm = ffm(grid)
  print (ffm.shape) ### should be [grid.shape[0], 2*L*rep_dim]

def unit_test_fourier_net():
  import ops, utils

  operator = ops.ParallelBeamGeometryOp(64, 60, 500)
  grid_params = {
    'angles': torch.cos(torch.tensor(operator.angles, dtype=torch.float32)),
    'num_detectors': operator.num_detectors
    }
  grid = utils.get_sino_mgrid(**grid_params)

  fn = FourierNet(in_features=2, hidden_features=256, hidden_blocks=2, out_features=1)
  fn(grid)

# if __name__=='__main__':
  # unit_test_ffm_dimension()
  # unit_test_fourier_net()
