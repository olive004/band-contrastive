import torch, torch.nn.functional as F
def ccl_loss(z: torch.Tensor, y: torch.Tensor, alpha: float=5.0, tau: float=0.1):
    """Continuous Contrastive Learning (weighted pairs by |y_i - y_j|)."""
    B = z.size(0)
    y = y.view(B,1)
    w = torch.exp(-alpha * (y - y.T).abs())
    z = F.normalize(z, dim=-1)
    sim = (z @ z.T) / tau
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~mask, float("-inf"))
    exp_sim = torch.exp(sim)
    num = (w * exp_sim * mask).sum(dim=1).clamp_min(1e-12)
    den = (exp_sim * mask).sum(dim=1).clamp_min(1e-12)
    return -(torch.log(num/den)).mean()
