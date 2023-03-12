from typing import Optional

import numpy as np
import torch
from PIL import ImageFilter
from scipy import signal
from torchvision.transforms.functional import to_tensor

from imgutils.data import load_image, ImageTyping


def np_hist(x, a_min: float = 0.0, a_max: float = 1.0, bins: int = 200):
    x = np.asarray(x)
    edges = torch.linspace(a_min, a_max, bins + 1).numpy()
    cnt, _ = np.histogram(x, bins=edges)

    return torch.from_numpy(cnt / cnt.sum())


def butterworth_filter(r, fc):
    w = fc / (len(r) / 2)  # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    sr = np.clip(signal.filtfilt(b, a, r), a_min=0.0, a_max=1.0)
    return torch.from_numpy(sr.copy())


def image_encode(image: ImageTyping, bins: int = 200, mf: Optional[int] = 5,
                 maxpixels: int = 20000, fc: Optional[int] = 30, normalize: bool = False):
    image = load_image(image, mode='RGB')
    if image.width * image.height > maxpixels:
        r = (image.width * image.height / maxpixels) ** 0.5
        new_width, new_height = map(lambda x: int(round(x / r)), image.size)
        image = image.resize((new_width, new_height))

    if mf is not None:
        image = image.filter(ImageFilter.MedianFilter(mf))
    image = image.convert('HSV')

    data = to_tensor(image)
    channels = [np_hist(data[i], bins=bins) for i in range(3)]
    if fc is not None:
        channels = [butterworth_filter(ch, fc) for ch in channels]

    dist = torch.stack(channels)
    assert dist.shape == (3, bins)

    if normalize:
        mean = torch.mean(dist, dim=1, keepdim=True)
        std = torch.std(dist, dim=1, keepdim=True)
        dist = (dist - mean) / std

    return dist
