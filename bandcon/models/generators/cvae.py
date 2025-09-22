import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from bandcon.models.mlp_vector import MLPVectorEncoder, VectorDecoder


def reparameterise(mu, logvar, seed: int, deterministic=False):
    """ We use exp(0.5*logvar) instead of std because it is more numerically stable
    and add the 0.5 part because std^2 = exp(logvar) """
    torch.manual_seed(seed)
    std = (0.5*logvar).exp()
    eps = torch.randn_like(std) if not deterministic else 0
    z = mu + eps * std
    return z


class VectorCVAE(pl.LightningModule):
    def __init__(self, model_cfg, x_dim: int, c_dim):
        super().__init__()
        self.z_dim = model_cfg.get("z_dim", 32)
        self.x_dim = x_dim
        self.c_dim = c_dim
        enc_hidden = model_cfg.get("encoder_hidden", [256, 256])
        dec_hidden = model_cfg.get("decoder_hidden", [256, 256])

        self.encoder = MLPVectorEncoder(x_dim + c_dim, enc_hidden, self.z_dim)
        self.decoder = VectorDecoder(self.z_dim + c_dim, dec_hidden, x_dim)
        self.beta = float(model_cfg.get("beta", 1.0))

    def encode(self, seed, x, c):
        mu, logvar = self.encoder(seed, torch.cat([x, c], dim=-1))
        z = reparameterise(mu, logvar, seed, deterministic=False)
        return z, mu, logvar

    def decode(self, seed, z, c):
        z_c = torch.cat([z, c], dim=-1)
        return self.decoder(seed, z_c)

    def forward(self, seed, x, c, return_embeddings=False):
        z, mu, logvar = self.encode(seed, x, c)
        recon = self.decode(seed, z, c)
        if return_embeddings:
            return recon, z, mu, logvar
        return recon
