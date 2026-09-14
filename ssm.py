# This is an implementation of an SSM, working baed on the ZOH Discretization results
# This also acts as a foundation for the Mamba module

import math
import torch.nn as nn
import torch
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
  def __init__(self, d_model, d_state, selective=True, stable_A=True):
    super().__init__()
    self.d_model = d_model
    self.d_state = d_state
    self.selective = selective
    self.stable_A = stable_A

    A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
    self.A_log = nn.Parameter(torch.log(A))            
    self.D = nn.Parameter(torch.ones(d_model))

    if selective:
        self.x_proj = nn.Linear(d_model, 2 * d_state + 1, bias=False)
    else:
        self.B_const = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))
        self.C_const = nn.Parameter(torch.randn(d_state) / math.sqrt(d_state))
    dt = torch.exp(torch.rand(d_model) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001))
    self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt))) 

  def forward(self, x, return_states=False):
    Bsz, L, D = x.shape
    A = -torch.exp(self.A_log)                                  

    if self.selective:
        proj = self.x_proj(x)                                   
        B, C, dt = proj.split([self.d_state, self.d_state, 1], dim=-1)
    else:
        B = self.B_const.expand(Bsz, L, -1)
        C = self.C_const.expand(Bsz, L, -1)
        dt = torch.zeros(Bsz, L, 1, device=x.device, dtype=x.dtype)
    delta = F.softplus(dt + self.dt_bias)                       

    h = torch.zeros(Bsz, D, self.d_state, device=x.device, dtype=x.dtype)
    ys, hs = [], []
    for t in range(L):
        dt_t = delta[:, t].unsqueeze(-1)                         
        A_bar = torch.exp(dt_t * A)                            
        B_bar = dt_t * B[:, t].unsqueeze(1)                      
        h = A_bar * h + B_bar * x[:, t].unsqueeze(-1)             
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))             
        if return_states:
            hs.append(h)
    y = torch.stack(ys, dim=1)                                  
    return (y, torch.stack(hs, dim=1)) if return_states else y

