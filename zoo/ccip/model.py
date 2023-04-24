import torch.nn
import torch.nn.functional as F
from PIL import Image
from torch import nn

from zoo.utils import get_testfile
from .backbone import get_backbone


class CCIPBatchMetrics(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.fc = nn.Linear(1, 2)

    def forward(self, x):  # x: BxN
        x = self.sim(x, x.unsqueeze(1))
        x = self.fc(x.unsqueeze(-1))
        return x


class CCIPFeature(torch.nn.Module):
    def __init__(self, name: str = "clip/ViT-B/32"):
        torch.nn.Module.__init__(self)
        self.backbone, self.preprocess = get_backbone(name)

    def forward(self, x):
        x = self.backbone(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class CCIP(torch.nn.Module):
    def __init__(self, name: str = "clip/ViT-B/32"):
        torch.nn.Module.__init__(self)
        self.feature = CCIPFeature(name)
        self.metrics = CCIPBatchMetrics()

    @property
    def preprocess(self):
        return self.feature.preprocess

    def forward(self, x):
        # x: BxCxHxW
        x = self.feature(x)  # BxF
        x = self.metrics(x)  # BxBx2
        return x


if __name__ == '__main__':
    image_files = [
        get_testfile('6124220.jpg'),
        get_testfile('6125785.jpg'),
        get_testfile('6125901.jpg'),
    ]

    model = CCIP()
    data = torch.stack([
        model.preprocess(Image.open(img))
        for img in image_files
    ])
    print(data.dtype, data.shape)

    print(F.softmax(model.forward(data), dim=-1))
