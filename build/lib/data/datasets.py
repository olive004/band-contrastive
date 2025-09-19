from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class DataCfg:
    name: str = "vectors_dummy"
    num_nodes: int = 8
    vector_dim: int = 28
    num_samples: int = 256
    noise: float = 0.05

class DummyVectorDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset to let reviewers run end-to-end on CPU.
    Produces vectors x and positive scalar labels y in (0,1).
    """
    def __init__(self, cfg):
        self.cfg = DataCfg(**cfg)
        rng = np.random.default_rng(0)
        self.x = rng.normal(0, 1, size=(self.cfg.num_samples, self.cfg.vector_dim)).astype(np.float32)
        # define y as a smooth function of x for a stable toy task
        self.y = (1/(1+np.exp(-self.x.mean(axis=1))) * (1 - self.cfg.noise) 
                  + rng.normal(0, self.cfg.noise, size=self.cfg.num_samples)).clip(1e-3, 1-1e-3).astype(np.float32)

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])
