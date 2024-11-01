import torch
from src.lit_model import LitModel
from src.datamodule.datamodule import ImageForgeryDatamMdule
from pytorch_lightning import Trainer

def test():
    path = "checkpoint/last.ckpt"
    data = "data/CASIA1"

    model = LitModel.load_from_checkpoint(path)
    data = ImageForgeryDatamMdule(folder_path=data)

    trainer = Trainer()
    trainer.test(model, datamodule=data)

if __name__ == "__main__":
    test()