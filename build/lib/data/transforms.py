import torch

def jitter(x, sigma=0.01):
    return x + sigma * torch.randn_like(x)
