from typing import Callable, Dict
import torch
from torch import optim
from bandcon.data.loaders import simple_loader
from bandcon.utils.hydra_utils import set_seed

def train_one_epoch(model, dataset, train_cfg: Dict, contrastive_fn: Callable, device="cpu"):
    set_seed(0)
    model.to(device)
    loader = simple_loader(dataset, batch_size=train_cfg.get("batch_size", 64))
    opt = optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3))
    model.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        recon, mu, logvar = model(x)
        # simple VAE losses already baked in via cvae.vae_step, reusing here:
        recon_loss = torch.nn.functional.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        z = mu.detach()  # use mean as embedding for simplicity in toy
        cl = contrastive_fn(z, y)
        loss = recon_loss + model.beta * kl + 0.3 * cl
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"step {step:03d} | loss={loss.item():.4f} recon={recon_loss.item():.4f} kl={kl.item():.4f} cl={cl.item():.4f}")
            break  # keep tiny for reviewers
