import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ssm import SelectiveSSM

class MambaBlock(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, selective=True):
        super().__init__()
        d_inner = expand * d_model
        self.d_conv = d_conv
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)   
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner)
        self.ssm = SelectiveSSM(d_inner, d_state, selective)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, u):
        x, gate = self.in_proj(self.norm(u)).chunk(2, dim=-1)
        x = F.pad(x.transpose(1, 2), (self.d_conv - 1, 0))
        x = self.conv(x).transpose(1, 2)
        x = self.ssm(F.silu(x)) * F.silu(gate)
        return u + self.out_proj(x)


class MambaLM(nn.Module):

    def __init__(self, vocab_size, d_model=64, n_layers=2, d_state=16, selective=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            MambaBlock(d_model, d_state, selective=selective) for _ in range(n_layers)
        )
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MambaLM(vocab_size=32, d_model=32, n_layers=2)
    out = m(torch.randint(0, 32, (2, 64)))
    print("logits:", out.shape)

    ssm = SelectiveSSM(d_model=16, d_state=8)
    _, h = ssm(torch.randn(1, 2000, 16), return_states=True)
    print("||h|| first/last:", h[0, 0].norm().item(), h[0, -1].norm().item())
