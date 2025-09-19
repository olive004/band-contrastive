import torch
from bandcon.losses.band_supcon import band_supcon_loss
from bandcon.losses.ccl import ccl_loss
from bandcon.losses.rnc import rnc_loss
from bandcon.losses.conr import conr_regularizer

def test_band_supcon_shapes():
    z = torch.randn(16, 8)
    y = torch.rand(16) * 0.9 + 0.05
    loss = band_supcon_loss(z, y, delta=0.10, tau=0.1)
    assert torch.isfinite(loss)

def test_ccl():
    z = torch.randn(16, 8)
    y = torch.rand(16)
    loss = ccl_loss(z, y, alpha=5.0, tau=0.1)
    assert torch.isfinite(loss)

def test_rnc():
    z = torch.randn(16, 8)
    y = torch.rand(16)
    loss = rnc_loss(z, y, margin=0.2)
    assert torch.isfinite(loss)

def test_conr():
    z = torch.randn(16, 8)
    y = torch.rand(16)
    reg = conr_regularizer(z, y, sigma=0.1)
    assert torch.isfinite(reg)
