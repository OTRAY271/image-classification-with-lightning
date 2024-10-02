import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    wandb_logger = WandbLogger(project=cfg.project_name)
    wandb_logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    model = hydra.utils.instantiate(cfg.litmodule)

    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    callbacks = []
    if "callbacks" in cfg.trainer:
        callbacks = hydra.utils.instantiate(cfg.trainer.callbacks)
    callbacks.append(ModelCheckpoint(f"{wandb_logger.experiment.dir}/ckpt"))
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    if "lr" not in cfg.litmodule:
        tuner = Tuner(trainer)
        tuner.lr_find(model=model, datamodule=datamodule)

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
