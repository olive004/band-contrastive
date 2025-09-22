

from functools import partial
import torch
import numpy as np


@torch.no_grad()
def accuracy(recon: torch.Tensor, x: torch.Tensor) -> float:
    preds = recon.argmax(dim=-1)
    correct = (preds == x).sum().item()
    total = x.numel()
    return correct / max(1, total)


@torch.no_grad()
def accuracy_regression(
    recon, x,
    threshold=0.1, **kwargs
) -> float:
    return torch.mean((torch.abs(x - recon) <= threshold).float())


def kl_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """ https://jaxopt.github.io/stable/auto_examples/deep_learning/haiku_vae.html """
    return 0.5 * torch.sum(-logvar - 1.0 + torch.exp(logvar) + torch.square(mu), dim=-1)


def l2_loss(weights, alpha):
    return alpha * (weights ** 2).mean()


def make_loss(cfg):
    
    loss_recon = cfg.loss.loss_recon
    if loss_recon == "mse":
        loss_recon_fn = torch.nn.MSELoss()
    elif loss_recon == "cross_entropy":
        loss_recon_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss_recon: {loss_recon}")

    use_l2_reg = cfg.loss.use_l2_reg
    use_kl_div = cfg.loss.use_kl_div
    use_contrastive_loss = cfg.loss.use_contrastive_loss
    if use_l2_reg:
        raise NotImplementedError("L2 regularization not implemented yet")
    if use_contrastive_loss:
        if cfg.cont_loss.name == 'band_supcon':
            from bandcon.losses.band_supcon import band_supcon_loss
            temperature = cfg.cont_loss.temperature
            delta = cfg.cont_loss.delta
            contrastive_loss_fn = partial(band_supcon_loss, tau=temperature, delta=delta)
        else:
            raise ValueError(f"Unsupported contrastive loss: {cfg.cont_loss.name}")
    else:
        contrastive_loss_fn = lambda *args, **kwargs: 0.0
    
    def loss_f(recon, x, c, z, mu, logvar, **model_call_kwargs):
        loss = loss_recon_fn(recon, x)
        
        # L2
        loss_l2 = None
        if use_l2_reg:
            loss_l2 = 0.0
            loss += loss_l2
            
        # KL divergence
        loss_kl = None
        if use_kl_div:
            loss_kl = kl_gaussian(mu, logvar).mean() * cfg.get("kl_weight", 1.0)
            loss += loss_kl
            
        # Contrastive loss
        loss_cl = None
        if use_contrastive_loss:
            loss_cl = contrastive_loss_fn(z=z, c=c)
            loss += loss_cl
        
        return loss, (loss_l2, loss_kl, loss_cl)

    return loss_f
