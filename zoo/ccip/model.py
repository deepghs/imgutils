import torch.nn
import torch.nn.functional as F
from PIL import Image
from torch import nn

from zoo.utils import get_testfile
from .backbone import get_backbone


class DiffMethod(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        """
        Bx1 --> Bx2
        """
        x = self.fc(x)
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
        self.backbone = CCIPFeature(name)
        self.diff = DiffMethod()
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)

    @property
    def preprocess(self):
        return self.backbone.preprocess

    def forward(self, x, y):
        x = self.backbone(x)
        y = self.backbone(y)
        dis = self.cos_sim(x, y)
        return self.diff(dis)


if __name__ == '__main__':
    image1 = Image.open(get_testfile('6124220.jpg'))
    image2 = Image.open(get_testfile('6125785.jpg'))

    model = CCIP()
    d1 = model.preprocess(image1).unsqueeze(0)
    d2 = model.preprocess(image2).unsqueeze(0)

    print(d1.shape, d1.dtype)
    print(d2.shape, d2.dtype)

    print(F.softmax(model.forward(d1, d2), dim=-1))
