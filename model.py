import torch
import torch.nn as nn
from typing import Optional

torch.set_default_dtype(torch.float32)

L_embed = 6

class NeRF(torch.nn.Module):
    """
    NeRF MLP model
    """
    def __init__(self, filter_size=128, L_embed=6):
        super(NeRF, self).__init__()
        self.layer1 = torch.nn.Linear(3 + 3*2*L_embed, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        self.relu = torch.nn.functional.relu
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def posenc(x):
    """
    Positional encoding
    """
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.0 ** i * x))
    return torch.cat(rets, dim=-1)

