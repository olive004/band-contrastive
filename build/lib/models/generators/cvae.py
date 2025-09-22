import torch, torch.nn as nn, torch.nn.functional as F
from band_contrastive.bandcon.models.mlp_vector import MLPVectorEncoder
from bandcon.models.decoders.vector_decoder import VectorDecoder
from bandcon.data.loaders import simple_loader

class VectorCVAE(nn.Module):
    def __init__(self, model_cfg, in_dim: int):
        super().__init__()
        self.z_dim = model_cfg.get("z_dim", 32)
        enc_hidden = model_cfg.get("encoder_hidden", [256,256])
        dec_hidden = model_cfg.get("decoder_hidden", [256,256])
        self.encoder = MLPVectorEncoder(in_dim, enc_hidden, self.z_dim)
        self.decoder = VectorDecoder(in_dim, dec_hidden, self.z_dim)
        self.beta = float(model_cfg.get("beta", 1.0))

    def encode(self, x):
        mu, logvar = self.encoder(x)
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_step(model, batch):
    x, y = batch
    recon, mu, logvar = model(x)
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + model.beta * kl
    return loss, {"recon": float(recon_loss.item()), "kl": float(kl.item())}

