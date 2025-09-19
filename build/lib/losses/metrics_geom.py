import torch

def alignment(z, y, delta=0.10):
    B = z.size(0)
    z = torch.nn.functional.normalize(z, dim=-1)
    d = ((z.unsqueeze(1) - z.unsqueeze(0))**2).sum(-1)
    return d.mean().item()

def uniformity(z):
    import torch
    z = torch.nn.functional.normalize(z, dim=-1)
    return torch.log(torch.exp(-2*((z.unsqueeze(1)-z.unsqueeze(0))**2).sum(-1)).mean()).item()
