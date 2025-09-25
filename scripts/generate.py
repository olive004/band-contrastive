import argparse, json, os
from bandcon.eval.reports import dummy_generate

from omegaconf import DictConfig
import numpy as np
import torch
from bandcon.data.datasets import DataNormaliser
from bandcon.models.utils import load_model, load_ckpt


def generate_samples(decoder, n_to_sample, z_dims, c_dims,
                     cond_min, cond_max, n_categories=None):

    # Sample noise vector
    h_dims = [n_to_sample, z_dims]
    z_noise = torch.randn(h_dims)

    # Sample conditions
    if n_categories is None:
        c = torch.empty([n_to_sample, c_dims], dtype=torch.float32).uniform_(cond_min, cond_max)
    else:
        import itertools
        c = np.array(list(itertools.product(
                *([np.linspace(cond_min, cond_max, n_categories).tolist()] * c_dims))))

    # Apply noise to decoder
    x_fake = decoder(0, torch.randn_like(torch.cat([z_noise, c], dim=-1)))

    return x_fake


def reverse_normalise(cfg_data: DictConfig, x_fake: torch.Tensor,
                      x_min, x_max) -> torch.Tensor:
    
    normaliser = DataNormaliser()
    
    if cfg_data.prep_x_negative:
        x_fake = normaliser.flip_negative(x_fake)
    if cfg_data.prep_x_logscale:
        x_fake = normaliser.inverse_log_scaling(x_fake)
    if cfg_data.prep_x_min_max:
        x_fake = normaliser.min_max_scaling(x_fake,
                                            range_prev=(0, 1),
                                            range_new=(x_min, x_max))

    return x_fake


def main(cfg: DictConfig):

    # Load trained model
    model, ds = load_model(cfg)

    # Generate samples
    x_fake = generate_samples(decoder=model.decoder, n_to_sample=cfg.eval.eval_n_to_sample,
                               z_dims=model.z_dim, c_dims=model.c_dim,
                               cond_min=cfg.eval.eval_cond_min, cond_max=cfg.eval.eval_cond_max,
                               n_categories=cfg.eval.eval_n_categories)
    
    # De-normalise samples from model outputs back to real data
    ckpt = load_ckpt(cfg)
    cfg_og = ckpt['cfg']
    x_sampled = reverse_normalise(cfg_og.data, x_fake, x_min=ds.x.min(), x_max=ds.x.max())
    
    np.save(x_sampled.detach().cpu().numpy(), cfg.eval.get("output_path", "x_sampled.npy"))    
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/dummy.ckpt")
    ap.add_argument("--target_y", type=float, default=0.75)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)
    out = dummy_generate(args.checkpoint, args.target_y, args.n)
    print(json.dumps(out, indent=2))
