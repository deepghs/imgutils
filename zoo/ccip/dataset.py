import glob
import os.path
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from imgutils.data import load_image
from .prob import get_reg_for_prob

TRAIN_TRANSFORM = [
    transforms.Resize(416),
    transforms.RandomRotation((-15, 15)),
    transforms.RandomCrop(384),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.10, 0.10),
    transforms.ToTensor(),
]
TEST_TRANSFORM = [
    transforms.Resize(416),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
]


class ImagesDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items: List[Tuple[str, int]] = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index) -> Tuple[Image.Image, int]:
        filename, idx = self.items[index]
        image = load_image(filename, mode='RGB')
        if self.transform:
            image = self.transform(image)

        return image, idx

    def split_dataset(self, test_prob: float = 0.2, train_transform=None, test_transform=None):
        total = len(self.items)
        test_ids = set(random.sample(list(range(total)), k=int(total * test_prob)))

        train_items, test_items = [], []
        for i, item in enumerate(self.items):
            if i in test_ids:
                test_items.append(item)
            else:
                train_items.append(item)

        return ImagesDataset(train_items, train_transform or self.transform), ImagesDataset(test_items, test_transform or self.transform)


class CCIPImagesDataset(ImagesDataset):
    def __init__(self, root_dir, transform=None):
        _ids, _maxid = {}, 0
        _items: List[Tuple[str, int]] = []
        for file in glob.glob(os.path.join(root_dir, '*', '*', '*.jpg')):
            dirname = os.path.normcase(os.path.normpath(os.path.dirname(os.path.abspath(file))))
            if dirname not in _ids:
                _ids[dirname] = _maxid
                _maxid += 1

            _items.append((file, _ids[dirname]))

        ImagesDataset.__init__(self, _items, transform)


class CharacterDataset(Dataset):
    def __init__(self, images_dataset: ImagesDataset, group_size: int = 100,
                 prob: float = 0.5, force_prob: bool = True):
        self.images_dataset = images_dataset
        self.groups: Dict[int, List[int]] = {}
        for i, (_, idx) in enumerate(self.images_dataset.items):
            if idx not in self.groups:
                self.groups[idx] = []
            self.groups[idx].append(i)

        self.group_size = group_size
        self._id_map = list(self.groups.keys())
        self._x_to_y, self._y_to_x = get_reg_for_prob(prob)
        self.force_prob = force_prob

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, item):
        idx = self._id_map[item]
        current_samples = self._x_to_y(self.group_size)
        if current_samples > len(self.groups[idx]):
            if self.force_prob:
                total_samples = self._y_to_x(len(self.groups[idx]))
                current_samples = self._x_to_y(total_samples)
                ex_samples = total_samples - current_samples
            else:
                current_samples = len(self.groups[idx])
                ex_samples = self.group_size - current_samples
        else:
            ex_samples = self.group_size - current_samples

        indices = []
        indices.extend(random.sample(self.groups[idx], k=current_samples))
        for _ in range(ex_samples):
            while True:
                t_idx = random.choice(list(self.groups.keys()))
                if t_idx != idx:
                    break
            indices.append(random.choice(self.groups[t_idx]))

        random.shuffle(indices)
        images, labels = [], []
        for i in indices:
            image, label = self.images_dataset[i]
            images.append(image)
            labels.append(label)

        return torch.stack(list(map(torch.as_tensor, images))), \
               torch.stack(list(map(torch.as_tensor, labels)))


class FastCharacterDataset(Dataset):
    def __init__(self, images_dataset: ImagesDataset, group_size: int = 100,
                 prob: float = 0.5, **kwargs):
        self.images_dataset = images_dataset

        groups: Dict[int, List[int]] = {}
        for i, (_, cid) in enumerate(self.images_dataset.items):
            if cid not in groups:
                groups[cid] = []
            groups[cid].append(i)
        self.groups = groups

        self.group_size = group_size
        self.prob = prob*2

    def reset(self):
        idxs = np.arange(0, len(self.images_dataset.items))
        np.random.shuffle(idxs)
        self.idxs = idxs[:-(len(idxs)%self.group_size)]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, cid = self.images_dataset[self.idxs[item]]
        n_same = int(self.prob) + int((self.prob-int(self.prob))<=((item%self.group_size+1)/self.group_size))

        image = [image]
        cid = [cid]
        if n_same>0:
            same_idxs = random.sample(self.groups[cid[0]], k=n_same)
            for idx in same_idxs:
                img_i, cid_i = self.images_dataset[idx]
                image.append(img_i)
                cid.append(cid_i)

        return image, cid

def char_collect_fn(batch):
    img_list, cid_list = [], []
    for data in batch:
        img_list.extend(data[0])
        cid_list.extend(data[1])
    return torch.stack(img_list), torch.tensor(cid_list)