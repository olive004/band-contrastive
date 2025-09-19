import hydra
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
