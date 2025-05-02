from omegaconf import OmegaConf, DictConfig
from ml_template.data import *
from models import *

import pytorch_lightning as pl
import hydra


@hydra.main(config_path="../../conf", config_name="config",version_base=None)
def train(cfg: DictConfig) -> None:
    print("Used config: ", OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    train()