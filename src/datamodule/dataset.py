from torch.utils.data.dataset import Dataset
from torchvision import transforms as tr
class ImageForgeryDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

        self.transform = tr.Compose([
            tr.Resize((256, 256)),
            tr.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.transform(image)
        return image, label