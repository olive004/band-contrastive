import torch

def conr_regularizer(z: torch.Tensor, y: torch.Tensor, sigma: float=0.1):
    """ConR regularizer: weighted L2 between embeddings of label-similar pairs."""
    B = z.size(0)
    y = y.view(B,1)
    w = torch.exp(-0.5*((y - y.T)/sigma)**2)
    d2 = ((z.unsqueeze(1) - z.unsqueeze(0))**2).sum(dim=-1)
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    return (w[mask] * d2[mask]).mean()
