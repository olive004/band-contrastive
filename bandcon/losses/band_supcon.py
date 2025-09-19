import torch
import torch.nn.functional as F
import math

def _cosine_sim(a, b, eps=1e-8):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.T

def band_supcon_loss(z: torch.Tensor, y: torch.Tensor, delta: float=0.10, tau: float=0.1):
    """Band-Supervised Contrastive loss with positives defined by a log-space ±delta band.
    Args:
        z: [B, d] latent embeddings
        y: [B] positive labels in (0, +inf), use log for scale-invariance
        delta: relative tolerance (e.g., 0.10 for ±10%)
        tau: temperature
    Returns:
        scalar loss tensor
    """
    B = z.size(0)
    logy = torch.log(y.clamp_min(1e-8)).view(B, 1)
    D = (logy - logy.T).abs()
    eps = math.log1p(delta)
    pos = (D <= eps) ^ torch.eye(B, dtype=torch.bool, device=z.device)
    sim = _cosine_sim(z, z) / tau
    # mask self
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~mask, float("-inf"))
    # numerator and denominator
    exp_sim = torch.exp(sim)
    num = (exp_sim * pos).sum(dim=1).clamp_min(1e-12)
    den = exp_sim.sum(dim=1).clamp_min(1e-12)
    valid = (pos.sum(dim=1) > 0)
    loss = -torch.log(num / den)[valid].mean() if valid.any() else torch.zeros([], device=z.device)
    return loss
