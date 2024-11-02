from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from PIL import Image, ImageFile
import os

from .dataset import ImageForgeryDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def extract_data(folder_path: str, type_path: str):
    data = []
    cls = os.listdir(os.path.join(folder_path, type_path))
    for c in cls:
        images_path = os.path.join(folder_path, type_path, c)
        for img in os.listdir(images_path):
            img_path = os.path.join(images_path, img)
            img = Image.open(img_path)
            label = 1 if c == "fake" else 0
            data.append((img, label))

    return data


class ImageForgeryDatamMdule(pl.LightningDataModule):
    def __init__(self,
                 folder_path,
                 num_workers: int = 4,
                 train_batch_size: int = 32,
                 val_batch_size: int = 16):
        super().__init__()
        self.folder_path = folder_path
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageForgeryDataset(
                extract_data(self.folder_path, "train")
            )
            self.val_dataset = ImageForgeryDataset(
                extract_data(self.folder_path, "val")
            )

        if stage == 'test' or stage is None:
            self.test_dataset = ImageForgeryDataset(
                extract_data(self.folder_path, "test")
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          persistent_workers=False,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          shuffle=False)
