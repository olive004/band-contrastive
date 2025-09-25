import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import jax


def simple_loader(ds, batch_size=64):
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def make_loaders(dataset, rng, train_frac=0.8, val_frac=0.1, batch_size=64):

    n = dataset.n_samples_max
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    randomised_indices = jax.random.choice(rng, np.arange(len(dataset.data)), [len(dataset.data)], replace=False)
    dataset.data = dataset.data.iloc[randomised_indices]
    g = torch.Generator().manual_seed(int(rng[0]))
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# def collate_as_graphs(batch):
#     """
#     Turn list of (x_nodes, x_edges, cond) into batched tensors.
#     Assumes fixed-size graphs (all same N).
#     """
#     x_nodes_list, x_edges_list, cond_list = zip(*batch)

#     if x_nodes_list[0] is not None:
#         x_nodes = torch.stack(x_nodes_list, dim=0)   # [B,N]
#     else:
#         x_nodes = None

#     x_edges = torch.stack(x_edges_list, dim=0)       # [B,N,N]

#     if cond_list[0] is not None:
#         cond = torch.stack(cond_list, dim=0)         # [B,c_dim]
#     else:
#         cond = None

#     return x_nodes, x_edges, cond
