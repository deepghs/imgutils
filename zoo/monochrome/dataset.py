import os
import random
from copy import deepcopy
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from copy import deepcopy
from tqdm.auto import tqdm
import random

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

TRANSFORM_val = transforms.Compose([
    transforms.Resize(450),
])

TRANSFORM2 = transforms.Compose([
    transforms.Resize(450),
    transforms.RandomCrop(400, padding=50, pad_if_needed=True, padding_mode='reflect'),
    transforms.RandomRotation((-180, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.10, 0.10),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

TRANSFORM2_val = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class MonochromeDataset(Dataset):
    def __init__(self, root_dir: str, bins: int = 180, fc: Optional[int] = 75, transform=TRANSFORM):
        self.root_dir = root_dir
        self.bins = bins
        self.fc = fc
        self.transform = transform
        self.samples = []
        self.pre_build = False

        mono_dir = os.path.join(root_dir, 'monochrome')
        for file_name in os.listdir(mono_dir):
            file_path = os.path.join(mono_dir, file_name)
            self.samples.append((file_path, 1))

        normal_dir = os.path.join(root_dir, 'normal')
        for file_name in os.listdir(normal_dir):
            file_path = os.path.join(normal_dir, file_name)
            self.samples.append((file_path, 0))

    def __len__(self):
        return len(self.samples)

    def pre_process(self, sample):
        image = Image.open(sample).convert('RGB')  # image must be rgb
        if self.transform:
            image = self.transform(image)
        image = image.convert('HSV')
        return image_encode(image, bins=self.bins, fc=self.fc, normalize=True)

    def cache_data(self, repeats=1):
        samples_build = []
        for sample, label in tqdm(self.samples*repeats):
            samples_build.append((self.pre_process(sample), label))
        self.samples = samples_build
        self.pre_build = True

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        if self.pre_build:
            return sample, label
        else:
            return self.pre_process(sample), label

class Monochrome2DDataset(MonochromeDataset):
    def __init__(self, root_dir: str, bins: int = 200, fc: Optional[int] = 50, transform=TRANSFORM2):
        super(Monochrome2DDataset, self).__init__(root_dir, bins, fc, transform)

    def pre_process(self, sample):
        image = Image.open(sample).convert('RGB')  # image must be rgb
        if self.transform:
            image = self.transform(image)
        return image

def random_split_dataset(dataset:MonochromeDataset, train_size, test_size, trans_val=TRANSFORM_val):
    train_data = deepcopy(dataset)
    random.shuffle(train_data.samples)
    all_samples = train_data.samples
    train_data.samples = train_data.samples[:train_size]

    test_data = dataset
    test_data.transform = trans_val
    print('pre-build testset')
    test_data.samples = all_samples[train_size:train_size+test_size]
    test_data.cache_data()

    return train_data, test_data