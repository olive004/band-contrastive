from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import partial
import math
import time
from pytorch_lightning.callbacks import ModelCheckpoint
import jax
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from bandcon.utils.hydra_utils import set_seed
from bandcon.data.datasets import DummyVectorDataset, make_dataset
from bandcon.data.loaders import make_loaders
from bandcon.models.utils import make_model
from bandcon.models.generators.cvae import VectorCVAE
from bandcon.train.loops import train_epoch
from bandcon.losses.general import make_loss
from bandcon.eval.metrics import simple_eval


def make_accuracy(name: str, thresh):
    if name == "regression":
        from bandcon.losses.general import accuracy_regression
        return partial(accuracy_regression, threshold=thresh)
    elif name == "regression_meandim":
        from bandcon.losses.general import accuracy_regression_meandim
        return partial(accuracy_regression_meandim, threshold=thresh)
    elif name == "classification":
        from bandcon.losses.general import accuracy
        return accuracy
    else:
        raise ValueError(f"Unsupported accuracy function: {name}")


def make_callbacks(cfg: DictConfig):

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)
    return callbacks


def make_optimiser(cfg: DictConfig, model):

    if cfg.optim.name == 'adam':
        optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, amsgrad=cfg.optim.amsgrad,
                                      weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.name == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9,
                                    weight_decay=cfg.optim.weight_decay)
    else:
        raise ValueError(f"Unsupported optimiser: {cfg.optim.name}")

    return optimiser


def make_scheduler(optimiser, cfg, steps_per_epoch):
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=1e-4, end_factor=1.0,
    #                                               total_iters=cfg.train.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=cfg.train.lr, steps_per_epoch=
                                                    steps_per_epoch, epochs=cfg.train.epochs, pct_start=0.3)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.train.epochs, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=cfg.train.epochs, T_mult=1, eta_min=1e-6)
    return scheduler


@dataclass
class TrainConfig:
    seed: int = 1
    epochs: int = 2000
    batch_size: int = 256
    lr: float = 1e-3
    save_model: bool = True
    threshold_early_val_acc: float = 0.98
    use_dropout: bool = False
    dropout_rate: float = 0.1
    clip_grad: bool = False

    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    accuracy_func: str = "regression"  # or "classification"
    accuracy_thresh: float = 0.1

    # Early stopping
    early_stop_metric: str = "val_loss"  # or "val_acc"
    patience: int = 100
    min_delta: float = 0.0  # improvement threshold

    # Checkpointing
    ckpt_dir: str = os.path.join(
        'band_contrastive', 'checkpoints', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    save_every_epoch: bool = False         # set True if you want all epochs
    best_ckpt_name: str = "best.pt"
    last_ckpt_name: str = "last.pt"

    # Logging
    check_val_every_n_epochs: int = 10
    log_every_n_epochs: int = 5


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        cfg: TrainConfig,
        accuracy_fn: nn.Module,
        loss_fn: nn.Module,
        optimiser: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg

        self.accuracy = accuracy_fn
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.scaler = torch.GradScaler('cuda', enabled=cfg.mixed_precision)
        
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        self.best_metric = math.inf if cfg.early_stop_metric == "val_loss" else -math.inf
        self.epochs_noimprove = 0
        
    def _apply_updates(self, loss):
        # ema_model = copy.deepcopy(model).eval()
        # for p, q in zip(model.parameters(), ema_model.parameters()):
        #     q.data.mul_(0.999).add_(p.data, alpha=0.001)
        
        self.optimiser.zero_grad(set_to_none=True)
        # loss.backward()
        # self.optimiser.step()
        self.scaler.scale(loss).backward()
        if self.cfg.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimiser)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

    def _step(self, seed, batch, train: bool = True) -> Dict[str, Any]:
        """
        Expects batch = (inputs, targets, *optional)
        For graph models you might pass a tuple; adapt here accordingly.
        """
        self.model.train(mode=train)
        x, c = batch[0], batch[1]  # adjust if your dataset returns more
        x = x.to(self.cfg.device)
        c = c.to(self.cfg.device)
        
        # Shuffle x and c equally using torch
        torch.manual_seed(seed)
        perm = torch.randperm(x.size(0), device=x.device)
        x = x[perm]
        c = c[perm]

        with torch.autocast('cuda', enabled=self.cfg.mixed_precision):
            recon, z, mu, logvar = self.model(
                seed, x, c, return_embeddings=True)
            loss, loss_logging = self.loss_fn(recon, x, c, z, mu, logvar)

        if train:
            self._apply_updates(loss)
            
        acc = self.accuracy(recon.detach(), x.detach())
        return {"loss": loss.item(), "acc": acc, 'loss_logging': loss_logging}

    @torch.no_grad()
    def _evaluate(self, rng, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        logged = {"loss": np.zeros(len(loader)),
                  "acc": np.zeros(len(loader)),
                  "l2": np.zeros(len(loader)),
                  "KL": np.zeros(len(loader)),
                  "CL": np.zeros(len(loader)),
                  }
        for ibatch, batch in enumerate(loader):
            x, c = batch[0], batch[1]
            x = x.to(self.cfg.device)
            c = c.to(self.cfg.device)
            rng, _ = jax.random.split(rng)
            seed = rng[0]
            recon, z, mu, logvar = self.model(
                seed, x, c, return_embeddings=True)
            loss, loss_extra = self.loss_fn(recon, x, c, z, mu, logvar)
            acc = self.accuracy(recon.detach(), x.detach())
            logged['loss'][ibatch] = loss.item()
            logged['acc'][ibatch] = acc
            for k, v in zip(['l2', 'KL', 'CL'], loss_extra):
                logged[k][ibatch] = v
        d = {'loss': logged['loss'].mean(), 'acc': logged['acc'].mean()}
        d.update({k: v.mean() for k, v in logged.items()})
        return d

    def _is_improved(self, val_metrics: Dict[str, float]) -> bool:
        if self.cfg.early_stop_metric == "val_loss":
            value = val_metrics["loss"]
            improved = (self.best_metric - value) > self.cfg.min_delta
            if improved:
                self.best_metric = value
            return improved
        else:  # "val_acc"
            value = val_metrics["acc"]
            improved = (value - self.best_metric) > self.cfg.min_delta
            if improved:
                self.best_metric = value
            return improved

    def _save_ckpt(self, name: str, epoch: int, val_metrics: Dict[str, float]):
        path = os.path.join(self.cfg.ckpt_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimiser_state": self.optimiser.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "val_metrics": val_metrics,
                # "dataset_in_dims": (self.train_loader.dataset.dataset.x_dim, self.train_loader.dataset.dataset.c_dim),
                "config": self.cfg.__dict__,
            },
            path,
        )
        return path

    def fit(self, rng):
        print(
            f"Starting training for {self.cfg.epochs} epochs on {self.cfg.device}.")
        history = []
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            # ---------- Train loop ----------
            logged = {"loss": np.zeros(len(self.train_loader)),
                      "acc": np.zeros(len(self.train_loader)),
                      "KL": np.zeros(len(self.train_loader))}
            for step, batch in enumerate(self.train_loader, 1):
                rng, _ = jax.random.split(rng)
                seed = rng[0]
                out = self._step(seed, batch, train=True)
                logged["loss"][step - 1] = out["loss"]
                logged["acc"][step - 1] = out["acc"]
                logged["KL"][step - 1] = out['loss_logging'][1].detach()

            # ---------- Validation ----------
            if self.val_loader is not None:
                val_metrics = self._evaluate(rng, self.val_loader)
            else:
                val_metrics = {"loss": float("nan"), "acc": float("nan")}

            # ---------- Logging ----------
            elapsed = time.time() - t0
            log_row = {
                "epoch": epoch,
                "train_loss": logged['loss'][-1],
                "train_acc": logged['acc'][-1],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "time_sec": elapsed,
                "lr": self.optimiser.param_groups[0]["lr"],
            }
            history.append(log_row)

            if epoch % self.cfg.log_every_n_epochs == 0:
                print(f"Epoch {epoch:03d} | "
                      f"train_loss last {logged['loss'][-1]:.4f} | train_acc max {logged['acc'].max():.4f} | "
                      f"loss KL last {logged['KL'][-1]:.4f} | "
                      f"val_loss {val_metrics['loss']:.4f} | val_acc {val_metrics['acc']:.4f} | "
                      f"{elapsed:.1f}s")

            # ---------- Checkpoints ----------
            # Always save "last"
            last_path = self._save_ckpt(
                self.cfg.last_ckpt_name, epoch, val_metrics)
            # Save best / early stopping
            improved = (self.val_loader is not None) and self._is_improved(
                val_metrics)
            if improved:
                best_path = self._save_ckpt(
                    self.cfg.best_ckpt_name, epoch, val_metrics)
                print(
                    f"Improved {self.cfg.early_stop_metric}. Saved best to: {best_path}")
                self.epochs_noimprove = 0
            else:
                self.epochs_noimprove += 1

            # Optional: save every epoch too
            if self.cfg.save_every_epoch:
                ep_path = self._save_ckpt(
                    f"epoch_{epoch:03d}.pt", epoch, val_metrics)

            # Early stopping check
            if (self.val_loader is not None) and (self.epochs_noimprove >= self.cfg.patience):
                print(
                    f"Early stopping triggered after {self.epochs_noimprove} bad epochs.")
                break

        print("Training done.")
        return history

    @torch.no_grad()
    def test(self, ckpt: Optional[str] = None) -> Dict[str, float]:
        if self.test_loader is None:
            print("No test_loader provided.")
            return {}
        if ckpt:
            state = torch.load(ckpt, map_location=self.cfg.device)
            self.model.load_state_dict(state["model_state"])
            print(f"Loaded checkpoint from {ckpt}")
        metrics = self._evaluate(self.test_loader)
        print(f"Test: loss {metrics['loss']:.4f} | acc {metrics['acc']:.4f}")
        return metrics


def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    ds = make_dataset(cfg.data)
    train_loader, val_loader, test_loader = make_loaders(
        ds, rng, cfg.data.train_frac, cfg.data.val_frac, cfg.train.batch_size)

    loss_fn = make_loss(cfg)
    accuracy_fn = make_accuracy(cfg.train.accuracy_func, cfg.train.accuracy_thresh)
    model = make_model(cfg.model, ds.x_dim, ds.c_dim)
    callbacks = make_callbacks(cfg)
    optimiser = make_optimiser(cfg, model)
    scheduler = make_scheduler(optimiser, cfg, len(train_loader))

    # from pytorch_lightning import Trainer
    # trainer = Trainer(strategy='auto',  # "ddp_find_unused_parameters_true",  # Needed to load old checkpoints
    #                   accelerator='gpu' if use_gpu else 'cpu',
    #                   devices=2 if use_gpu else 1,
    #                   max_epochs=cfg.train.epochs,
    #                   callbacks=callbacks,
    #                   fast_dev_run=cfg.name == 'debug',
    #                   check_val_every_n_epoch=cfg.train.check_val_every_n_epochs,
    #                   enable_progress_bar=True,
    #                   gradient_clip_val=cfg.train.clip_grad,
    #                   log_every_n_epochs=cfg.train.log_every_n_epochs if cfg.name != 'debug' else 1,
    #                   logger=[])
    # trainer.fit(model, datamodule=ds)  # , ckpt_path=cfg.resume)
    # trainer.test(model, datamodule=ds)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=TrainConfig(**cfg.train),
        accuracy_fn=accuracy_fn,
        loss_fn=loss_fn,
        optimiser=optimiser,
        scheduler=scheduler)

    trainer.fit(rng)


def example(cfg: DictConfig):
    """ Minimal CPU-friendly toy training loop to prove wiring """

    set_seed(cfg.get("seed", 0))
    ds = DummyVectorDataset(cfg.data)
    model = VectorCVAE(cfg.model, ds.vector_dim)

    # one tiny epoch on CPU
    from bandcon.losses.band_supcon import band_supcon_loss
    train_epoch(model, ds, cfg.train, contrastive_fn=band_supcon_loss,
                device=cfg.get("device", "cpu"))

    # save a tiny checkpoint substitute
    os.makedirs("outputs", exist_ok=True)
    torch.save({"state": "dummy", "z_dim": model.z_dim}, "outputs/dummy.ckpt")

    # run a tiny eval
    res = simple_eval([0.3, 0.5, 0.7], tolerances=[0.10, 0.05])
    print("EVAL:", res)
