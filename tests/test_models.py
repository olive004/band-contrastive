import torch
from bandcon.models.generators.cvae import VectorCVAE

def test_vector_cvae_forward():
    in_dim = 28
    model = VectorCVAE({"z_dim": 16, "encoder_hidden":[64], "decoder_hidden":[64], "beta":1.0}, in_dim)
    x = torch.randn(4, in_dim)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape[-1] == 16
