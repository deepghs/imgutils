from functools import lru_cache
from typing import List

import lpips
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from torchvision import transforms
from tqdm.auto import tqdm

from .models import LPIPSDiff, LPIPSFeature

_TRANSFORM = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
])


class LPIPSSimilarityDetection:
    def __init__(self, thr=0.45):
        self.model = lpips.LPIPS(net='alex', spatial=False)
        self._get_feat = LPIPSFeature(self.model)
        self._diff = LPIPSDiff(self.model)

        self.thr = thr

    @torch.no_grad()
    def diff(self, im1, im2):
        im1 = _TRANSFORM(im1.convert('RGB')).unsqueeze(0)
        im2 = _TRANSFORM(im2.convert('RGB')).unsqueeze(0)
        feat1 = self._get_feat(im1)
        feat2 = self._get_feat(im2)
        scores = self._diff(*feat1, *feat2)
        return scores.item()

    @torch.no_grad()
    def cluster(self, images: List[Image.Image]) -> List[int]:
        n = len(images)
        img_list = [_TRANSFORM(img.convert('RGB')).unsqueeze(0) for img in tqdm(images, leave=False)]
        feat_list = [self._get_feat(img) for img in tqdm(img_list, leave=False)]

        progress = tqdm(total=n * (n + 1) // 2, leave=False)

        @lru_cache(maxsize=n * (n + 1) // 2)
        def _cached_metric(x, y):
            result = self._diff(*feat_list[x], *feat_list[y]).item()
            progress.update(1)
            return result

        def img_sim_metric(x, y):
            x, y = int(min(x, y)), int(max(x, y))
            return _cached_metric(x, y)

        samples = np.array(range(n)).reshape(-1, 1)
        clustering = DBSCAN(eps=self.thr, min_samples=2, metric=img_sim_metric).fit(samples)
        progress.close()
        return clustering.labels_
