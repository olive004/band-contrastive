import hydra
from omegaconf import DictConfig
from bandcon.cli.eval import main as eval_main

@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def _main(cfg: DictConfig):
    eval_main(cfg)

if __name__ == "__main__":
    _main()
