
from sconf import Config
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
from src.lit_model import LitModel
from src.datamodule.datamodule import ImageForgeryDatamMdule



def train(config):
    pl.seed_everything(config.seed_everything, workers=True)

    model_module = LitModel(
        num_classes=config.model.num_classes,
        d_model=config.model.d_model,
        # spatial transformer
        requires_grad=config.model.requires_grad,
        drop_rate=config.model.drop_rate,
        proj_drop_rate=config.model.proj_drop_rate,
        attn_drop_rate=config.model.attn_drop_rate,
        drop_path_rate=config.model.drop_path_rate,

        # frequency transformer
        growth_rate=config.model.growth_rate,
        num_layers=config.model.num_layers,

        # training
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
        patience=config.model.patience
    )

    data_module = ImageForgeryDatamMdule(
        folder_path=config.data.folder_path,
        num_workers=config.data.num_workers,
        train_batch_size=config.data.train_batch_size,
        val_batch_size=config.data.val_batch_size
    )

    wandb_logger = WandbLogger(name=config.wandb.name,
                    project=config.wandb.project,
                    log_model=config.wandb.log_model,
                    config=dict(config),
                    )
    wandb_logger.watch(model_module,
                 log="all",
                 log_freq=50,
                 )

    lasted_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoint",
        save_last=True,
    )


    lr_callback = pl.callbacks.LearningRateMonitor(
        logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=config.trainer.callbacks[1].init_args.save_top_k,
        monitor=config.trainer.callbacks[1].init_args.monitor,
        mode=config.trainer.callbacks[1].init_args.mode,
        filename=config.trainer.callbacks[1].init_args.filename)

    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        max_epochs=config.trainer.max_epochs,
        deterministic=config.trainer.deterministic,

        callbacks=[lr_callback, checkpoint_callback, lasted_checkpoint_callback],
        logger=wandb_logger,

        log_every_n_steps=10
    )
    trainer.fit(model_module, data_module)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    train(config)