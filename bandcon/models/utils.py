

import torch.nn as nn
import torch.nn.init as init


# Sampling helpers, masks, etc. (stubs)
def sample_latent(z_dim: int, n: int):
    import torch
    return torch.randn(n, z_dim)


def init_weights(model: nn.Module, method: str = "xavier", bias_const: float = 0.0):
    
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if method == "xavier":
                init.xavier_uniform_(m.weight)
            elif method in ["kaiming", "he"]:
                init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
            elif method == "normal":
                init.normal_(m.weight, mean=0.0, std=0.02)
            elif method == "uniform":
                init.uniform_(m.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(f"Unknown init method: {method}")
            if m.bias is not None:
                init.constant_(m.bias, bias_const)
                
            # Initialize biases to zero, if they exist (nn.Linear has bias by default)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            init.constant_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Embedding):
            print('\n\nInitialising embedding layer with normal distribution (mean=0.0, std=0.02)\n\n')
            init.normal_(m.weight, mean=0.0, std=0.02)
            
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
    model.encoder = init_weights(model.encoder, model_cfg.enc_init)
    model.decoder = init_weights(model.decoder, model_cfg.dec_init)
    return model
