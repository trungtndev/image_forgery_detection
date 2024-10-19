from sconf import Config
import argparse
import pytorch_lightning as pl
from src.lit_model import LitModel
from src.datamodule.datamodule import ImageForgeryDatamMdule


def train(config):
    pl.seed_everything(config.seed_everything, workers=True)
    path = "checkpoint/epoch=5-step=3786.ckpt"

    model_module = LitModel.load_from_checkpoint(path)

    data_module = ImageForgeryDatamMdule(
        folder_path=config.data.folder_path,
        num_workers=config.data.num_workers,
        train_batch_size=config.data.train_batch_size,
        val_batch_size=config.data.val_batch_size
    )


    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        deterministic=config.trainer.deterministic,
    )
    trainer.test(model_module, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    train(config)
