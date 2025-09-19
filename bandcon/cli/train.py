from omegaconf import DictConfig
from bandcon.utils.hydra_utils import set_seed
from bandcon.data.datasets import DummyVectorDataset
from bandcon.models.generators.cvae import VectorCVAE
from bandcon.train.loops import train_one_epoch
from bandcon.losses.band_supcon import band_supcon_loss
from bandcon.eval.metrics import simple_eval

def main(cfg: DictConfig):
    # Minimal CPU-friendly toy training loop to prove wiring
    set_seed(cfg.get("seed", 0))
    ds = DummyVectorDataset(cfg.data)
    model = VectorCVAE(cfg.model, ds.vector_dim)
    # one tiny epoch on CPU
    train_one_epoch(model, ds, cfg.train, contrastive_fn=band_supcon_loss, device=cfg.get("device","cpu"))
    # save a tiny checkpoint substitute
    import torch, os
    os.makedirs("outputs", exist_ok=True)
    torch.save({"state":"dummy", "z_dim": model.z_dim}, "outputs/dummy.ckpt")
    # run a tiny eval
    res = simple_eval([0.3,0.5,0.7], tolerances=[0.10,0.05])
    print("EVAL:", res)
