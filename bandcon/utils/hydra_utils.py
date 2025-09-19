import os, random, numpy as np, torch

def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
