from omegaconf import DictConfig
from typing import Dict, Tuple
import numpy as np
import torch
from bandcon.data.datasets import make_dataset, BasicDataset, DataNormaliser
from bandcon.eval.metrics import simple_eval
from bandcon.models.utils import make_model


def load_ckpt(cfg: DictConfig) -> Dict:

    ckpt_path = cfg.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt


def locate_model_cls(name):
    if name == 'cvae_vector':
        from bandcon.models.generators.cvae import VectorCVAE
        return VectorCVAE
    else:
        raise NotImplementedError(
            f'Model {name} not found or implemented yet.')


def strip_dataparallel_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handle checkpoints saved with DataParallel / DDP (keys start with "module.")
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_model(cfg: DictConfig) -> Tuple[torch.nn.Module, BasicDataset]:
    
    ckpt = load_ckpt(cfg)
    state = ckpt['model_state']
    cfg_og = ckpt['cfg']

    torch.manual_seed(cfg_og.get("seed", 0))
    ds = make_dataset(cfg_og.data)
    model = make_model(cfg_og.model, ds.x_dim, ds.c_dim)

    # Support both {"model_state": ...} or raw state_dict
    if isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
    elif isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
        state_dict = state
    else:
        raise RuntimeError(
            f"Checkpoint format not recognized: keys={list(state.keys())[:5]}")

    state_dict = strip_dataparallel_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] Missing keys: {missing}")
    if unexpected:
        print(f"[load_model] Unexpected keys: {unexpected}")

    return model, ds


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
                      x_min, x_max):
    
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

    fn_simulation_settings: 'band_contrastive/configs/simulation_settings.json'

    # Load trained model
    model, ds = load_model(cfg)

    # Generate samples
    x_fake = generate_samples(decoder=model.decoder, n_to_sample=cfg.eval.eval_n_to_sample,
                               z_dims=model.z_dim, c_dims=model.c_dim,
                               cond_min=cfg.eval.eval_cond_min, cond_max=cfg.eval.eval_cond_max,
                               n_categories=cfg.eval.eval_n_categories)
    
    # De-normalise samples from model outputs back to real data
    cfg_og = cfg['cfg']
    x_sampled = reverse_normalise(cfg_og.data, x_fake, x_min=ds.x.min(), x_max=ds.x.max())
    
    np.save(x_sampled.detach().cpu().numpy(), cfg.eval.get("output_path", "x_sampled.npy"))    
    

def example(cfg: DictConfig):
    res = simple_eval(cfg.eval.get("targets", [0.3, 0.5, 0.7]), cfg.eval.get(
        "tolerances", [0.10, 0.05]))
    print("EVAL:", res)
