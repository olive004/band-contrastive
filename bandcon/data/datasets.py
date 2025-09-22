from dataclasses import dataclass
from typing import Tuple
from omegaconf import DictConfig
from bandcon.data.processing.physical import embellish_data
from bandcon.data.processing.filter import filter_invalids
from torch_geometric.data.lightning import LightningDataset
import numpy as np
import pandas as pd
import torch
import os


@dataclass
class DataCfg:
    name: str = "vectors_dummy"
    n_nodes: int = 8
    vector_dim: int = 28
    n_samples_max: int = 256
    noise: float = 0.05


class BasicDataset(LightningDataset):
    """ Basic dataset wrapping input and label pairs.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data = np.array([])
        self.n_samples_max = cfg.get("n_samples_max", 256)

    def __len__(self):
        return np.min([self.n_samples_max, len(self.data)])

    # def __getitem__(self, idx):
    #     d = self.data[idx]
    #     x_nodes = d.get("nodes")
    #     x_edges = d["edges"]
    #     cond = d.get("cond", None)
    #     return x_nodes, x_edges, cond

    # def __getitem__(self, idx):
    #     return torch.from_numpy(self.x[idx]), torch.from_numpy(self.c[idx])


def load_data(cfg):

    if os.path.splitext(cfg.data_path)[1] == '.json':
        data = pd.read_json(cfg.data_path)
    elif os.path.splitext(cfg.data_path)[1] == '.csv':
        data = pd.read_csv(cfg.data_path)
    else:
        raise ValueError(f"Unsupported file extension: {cfg.data_path}")

    data = embellish_data(data)
    x_cols = list([c for c in data.columns if c.startswith(cfg.x_type)])
    data = filter_invalids(data, x_cols, cfg)
    # df = reduce_repeat_samples(
    #     df, cfg.cols_x, n_same_circ_max=cfg.filter_settings.filt_n_same_x_max, nbin=cfg.filter_settings.filt_n_same_x_max_bins)

    return data, x_cols


def calc_minmax(data, min_val, scale, max_range, min_range):
    return ((data - min_val) / scale) * \
        (max_range - min_range) + min_range


class DataNormaliser:
    
    def __init__(self):
        self.metadata = {}
        
    def _fetch_metadata(self, key, default, col=None):
        try:
            if col:
                return self.metadata.get(col, {}).get(key, default)
            return self.metadata.get(key, default)
        except KeyError:
            raise KeyError(f"Metadata for {key} not found. Please run normalisation without use_precomputed first.")
    
    def min_max_scaling(
        self,
        data,
        feature_range: Tuple[float, float] = (0, 1),
        col=None,
        use_precomputed: bool = False,
        tiny: float = 1e-6,
    ):
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        scale = max_val - min_val

        if use_precomputed:
            min_val = self._fetch_metadata('min_val', min_val, col)
            scale = self._fetch_metadata('scale', scale, col)

        # Prevent division by zero
        scale = np.where(scale == 0, tiny, scale)

        # Map to desired feature range
        min_range, max_range = feature_range
        scaled = calc_minmax(data, min_val, scale, max_range, min_range)

        if not use_precomputed:
            # Store for later use
            k = {
                'min_val': min_val,
                'scale': scale,
                'feature_range': feature_range
            }
            if col:
                self.metadata.setdefault(col, {}).update(k)
            else:
                self.metadata.update(k)

        return scaled
    
    def flip_negative(self, data, **kwargs):
        return - data
    
    def log_scaling(self, data, **kwargs):
        return np.log10(data)
    

def prep(data, x_cols, c_cols, cfg, normaliser: DataNormaliser):
    
    def apply_c_f(c, normaliser_f):
        for cdim in range(c.shape[-1]):
            c[..., cdim] = normaliser_f(c[..., cdim], col=c_cols[cdim])
        return c
    
    x = data[x_cols].values
    c = data[c_cols].values

    if cfg.prep_x_negative:
        x = normaliser.flip_negative(x)
    if cfg.prep_y_negative:
        c = apply_c_f(c, normaliser.flip_negative)
    if cfg.prep_x_logscale:
        x = normaliser.log_scaling(x)
    if cfg.prep_y_logscale:
        c = apply_c_f(c, normaliser.log_scaling)
    if cfg.prep_x_min_max:
        x = normaliser.min_max_scaling(x)
    if cfg.prep_y_min_max:
        c = apply_c_f(c, normaliser.min_max_scaling)
    
    return x, c


class RNADataset(BasicDataset):
    """ Dataset wrapping input and label pairs for RNA data.
    """

    def __init__(self, cfg_data: DictConfig):
        super().__init__(cfg_data)

        self.data, self.x_cols = load_data(self.cfg)
        self.normaliser = DataNormaliser()
        x, c = prep(self.data, self.x_cols, self.cfg.objective, self.cfg, self.normaliser)
        assert len(x) == len(c), "x and c must have the same length"
        self.x = x.astype(np.float32)
        self.c = c.astype(np.float32)
        self.vector_dim = x.shape[-1] + c.shape[-1]
        self.x_dim = x.shape[-1] 
        self.c_dim = c.shape[-1]
        self.n_samples_max = int(
            min(len(self.x), self.cfg.get("n_samples_max", 256)))

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.c[idx])

class DummyVectorDataset(BasicDataset):
    """ Small synthetic dataset to let reviewers run end-to-end on CPU.
    Produces vectors x and positive scalar labels y in (0,1).
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = DataCfg(**cfg)
        rng = np.random.default_rng(cfg.get("seed", 0))
        self.vector_dim = self.cfg.vector_dim
        self.x = rng.normal(0, 1, size=(
            self.cfg.n_samples_max, self.cfg.vector_dim)).astype(np.float32)

        # Define y as a smooth function of x for a stable toy task
        self.y = (1/(1+np.exp(-self.x.mean(axis=1))) * (1 - self.cfg.noise)
                  + rng.normal(0, self.cfg.noise, size=self.cfg.n_samples_max)).clip(1e-3, 1-1e-3).astype(np.float32)

    def __len__(self):
        return np.min([self.n_samples_max, len(self.x)])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])
