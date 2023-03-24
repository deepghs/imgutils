import os
import random
from copy import deepcopy
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm.auto import tqdm

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

TRANSFORM_VAL = transforms.Compose([
    transforms.Resize(450),
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

    def get_hist(self, sample):
        image = Image.open(sample).convert('RGB')  # image must be rgb
        if self.transform:
            image = self.transform(image)
        image = image.convert('HSV')
        return image_encode(image, bins=self.bins, fc=self.fc, normalize=True)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        if self.pre_build:
            return sample, label
        else:
            return self.get_hist(sample), label


def random_split_dataset(dataset: MonochromeDataset, train_size, test_size):
    train_data = deepcopy(dataset)
    random.shuffle(train_data.samples)
    all_samples = train_data.samples
    train_data.samples = train_data.samples[:train_size]

    test_data = dataset
    test_data.transform = TRANSFORM_VAL
    samples_build = []
    # print('pre-build testset')
    for sample, label in tqdm(all_samples[train_size:train_size + test_size]):
        samples_build.append((test_data.get_hist(sample), label))
    test_data.samples = samples_build
    test_data.pre_build = True

    return train_data, test_data
