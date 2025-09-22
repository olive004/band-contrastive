import hydra
from omegaconf import DictConfig
from bandcon.cli.train import main as train_main


@hydra.main(version_base=None, config_path="../configs", config_name="vectors")
# @hydra.main(version_base=None, config_path="../configs", config_name="exp/vector_cvae_band")
def _main(cfg: DictConfig):
    train_main(cfg)

if __name__ == "__main__":
    _main()
