import torch
import torch.nn.functional as F

def rnc_loss(z: torch.Tensor, y: torch.Tensor, margin: float=0.2):
    """Rank-N-Contrast: simple triplet version for toy runs.
    For each anchor, choose the closest and farthest label neighbors in the batch.
    """
    B = z.size(0)
    y = y.view(B,1)
    with torch.no_grad():
        dlabel = (y - y.T).abs()
        closest = torch.topk(dlabel + torch.eye(B, device=z.device)*1e9, k=1, largest=False).indices.squeeze(-1)
        farthest = torch.topk(dlabel, k=1, largest=True).indices.squeeze(-1)
    z = F.normalize(z, dim=-1)
    def d(a,b): return 1 - (a*b).sum(dim=-1)
    zi = z
    zj = z[closest]
    zk = z[farthest]
    losses = torch.clamp(margin + d(zi,zj) - d(zi,zk), min=0.0)
    return losses.mean()
