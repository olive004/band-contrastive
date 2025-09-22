import torch.nn as nn
from typing import List


def create_layers(in_dim: int, output_sizes: List[int], z_dim: int, activation_final=None):
    l = []
    prev = output_sizes[0]
    l += [nn.Linear(in_dim, prev), nn.LeakyReLU()]
    for i, output_size in enumerate(output_sizes[1:]):
        
        l += [nn.Linear(prev, output_size), nn.LeakyReLU()]
        prev = output_size

    l += [nn.Linear(prev, z_dim)]
    if activation_final is not None:
        l.append(activation_final)
        
    return l


class MLPVector(nn.Module):
    def __init__(self, in_dim: int, hidden, z_dim: int, activation_final=None):
        super().__init__()
        self.activation_final = activation_final or nn.LeakyReLU()

        # layers = []
        # last = in_dim
        # for h in hidden:
        #     layers += [nn.Linear(last, h), nn.LeakyReLU()]
        #     last = h
        layers = create_layers(in_dim, hidden, z_dim, self.activation_final)
        self.net = nn.Sequential(*layers)
        
    def forward(self, key, input):

        return self.net(input)
    
    
class MLPVectorEncoder(MLPVector):
    def __init__(self, in_dim: int, hidden, z_dim: int, activation_final=None):
        super().__init__(in_dim, hidden, z_dim, activation_final)
        last = hidden[-1] if len(hidden) > 0 else in_dim
        self.mu = nn.Linear(last, z_dim)
        self.logvar = nn.Linear(last, z_dim)

    def forward(self, key, input):
        h = self.net(input)
        return self.mu(h), self.logvar(h)


class VectorDecoder(MLPVector):
    def __init__(self, z_dim: int, hidden, out_dim: int, activation_final=None):
        super().__init__(z_dim, hidden, out_dim, activation_final)
