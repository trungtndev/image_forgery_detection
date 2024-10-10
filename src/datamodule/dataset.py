from torch.utils.data.dataset import Dataset
from torchvision import transforms as tr



class ImageForgeryDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

        self.transform = tr.Compose([
            tr.Resize((224, 224)),
            tr.ToTensor(),
            # tr.Normalize(mean=[0.485, 0.456, 0.406],
            #              std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.transform(image)
        return image, label
