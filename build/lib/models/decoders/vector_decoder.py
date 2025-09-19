import torch, torch.nn as nn

class VectorDecoder(nn.Module):
    def __init__(self, out_dim: int, hidden, z_dim: int):
        super().__init__()
        layers = []
        last = z_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # For toy: treat outputs as mean of Gaussian; loss will be MSE
        return self.net(z)
