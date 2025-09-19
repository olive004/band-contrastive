import torch, torch.nn as nn

class MLPVectorEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden, z_dim: int):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(last, z_dim)
        self.logvar = nn.Linear(last, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)
