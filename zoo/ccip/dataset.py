import glob
import os.path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset

from imgutils.data import load_image


class CCIPDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        _ids, _maxid = {}, 0
        self.items: List[Tuple[str, int]] = []
        self.transform = transform

        for file in glob.glob(os.path.join(root_dir, '*', '*', '*.jpg')):
            dirname = os.path.normcase(os.path.normpath(os.path.dirname(os.path.abspath(file))))
            if dirname not in _ids:
                _ids[dirname] = _maxid
                _maxid += 1

            self.items.append((file, _ids[dirname]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index) -> Tuple[Image.Image, int]:
        filename, idx = self.items[index]
        image = load_image(filename, mode='RGB')
        if self.transform:
            image = self.transform(image)

        return image, idx
