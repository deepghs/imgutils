import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from .encode import image_encode

TRANSFORM = transforms.Compose([
    transforms.Resize(900),
    transforms.RandomCrop(800, padding=150, pad_if_needed=True, padding_mode='reflect'),
    transforms.RandomRotation((-180, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.10, 0.10),
    transforms.Resize(450),
])


class ImageDirectoryDataset(Dataset):
    def __init__(self, root_dir, label: int = 1, bins: int = 200, fc: Optional[int] = 50, transform=TRANSFORM):
        self.root_dir = root_dir
        self.label = label
        self.bins = bins
        self.fc = fc
        self.transform = transform
        self.samples = []
        for file_name in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file_name)
            self.samples.append(file_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        image = Image.open(file_path).convert('HSV')
        if self.transform:
            image = self.transform(image)
        return image_encode(image, bins=self.bins, fc=self.fc, normalize=True), torch.tensor(self.label)


class MonochromeDataset(Dataset):
    def __init__(self, root_dir: str, bins: int = 200, fc: Optional[int] = 50, transform=TRANSFORM):
        self.monochrome = ImageDirectoryDataset(os.path.join(root_dir, 'monochrome'), 1, bins, fc, transform)
        self.normal = ImageDirectoryDataset(os.path.join(root_dir, 'normal'), 0, bins, fc, transform)

    def __len__(self):
        return len(self.monochrome) + len(self.normal)

    def __getitem__(self, idx):
        if idx < len(self.monochrome):
            return self.monochrome[idx]
        else:
            return self.normal[idx - len(self.monochrome)]
