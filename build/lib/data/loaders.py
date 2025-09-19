import torch

def simple_loader(ds, batch_size=64):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
