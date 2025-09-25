

from omegaconf import DictConfig
from typing import Dict, Tuple
import torch
import torch.nn as nn
from bandcon.data.datasets import make_dataset, BasicDataset


# Sampling helpers, masks, etc. (stubs)
def sample_latent(z_dim: int, n: int):
    return torch.randn(n, z_dim)


def init_weights(seed: int, model: nn.Module, method: str = "xavier", bias_const: float = 0.0):
    
    torch.manual_seed(seed)
    
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method in ["kaiming", "he"]:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
            elif method == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif method == "uniform":
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(f"Unknown init method: {method}")
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_const)
                
            # Initialize biases to zero, if they exist (nn.Linear has bias by default)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Embedding):
            print('\n\nInitialising embedding layer with normal distribution (mean=0.0, std=0.02)\n\n')
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    return model


def make_model(model_cfg, x_dim: int, c_dim: int):
    if model_cfg.name == 'cvae_vector':
        from bandcon.models.generators.cvae import VectorCVAE
        model = VectorCVAE(model_cfg, x_dim, c_dim)
    elif model_cfg.name == 'diffusion_vector':
        from bandcon.models.generators.diffusion import ConditionalDDPM
        model = ConditionalDDPM(model_cfg, x_dim, c_dim)
    elif model_cfg.name == 'diffusion_graph':
        # from bandcon.models.generators.diffusion import DiGressConditional
        raise NotImplementedError(f"Model {model_cfg.name} not implemented")
        # return DiGressConditional(model_cfg, vector_dim)
    else:
        raise NotImplementedError(f"Model {model_cfg.name} not implemented")
    model.encoder = init_weights(model_cfg.seed, model.encoder, model_cfg.enc_init)
    model.decoder = init_weights(model_cfg.seed, model.decoder, model_cfg.dec_init)
    return model


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
