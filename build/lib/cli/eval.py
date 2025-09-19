from omegaconf import DictConfig
from bandcon.eval.metrics import simple_eval

def main(cfg: DictConfig):
    res = simple_eval(cfg.eval.get("targets",[0.3,0.5,0.7]), cfg.eval.get("tolerances",[0.10,0.05]))
    print("EVAL:", res)
