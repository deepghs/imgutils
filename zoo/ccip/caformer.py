import torch.nn
from torchvision.transforms import InterpolationMode, Compose, Resize, CenterCrop, ToTensor, Normalize

from .attention_pool import AttentionPool2d
from ..monochrome.metaformer import CAFormerBuilder


class CaformerBackbone(torch.nn.Module):
    def __init__(self, input_resolution: int = 384, heads: int = 32, out_dims: int = 1024, **kwargs):
        torch.nn.Module.__init__(self)
        self.input_resolution = input_resolution
        self.caformer = CAFormerBuilder(**kwargs)()
        self.attnpool = AttentionPool2d(self.input_resolution // 32, self.caformer.output_dim, heads, out_dims)

    def _get_cnn_result(self, x):
        for i in range(self.caformer.num_stage):
            x = self.caformer.downsample_layers[i](x)
            x = self.caformer.stages[i](x)

        x = x.permute(0, 3, 1, 2)  # BxHxWxC --> BxCxHxW
        return x

    def forward(self, x):
        x = self._get_cnn_result(x)
        x = self.attnpool(x)
        return x


def get_caformer(input_resolution: int = 224, heads: int = 32, feat_dims: int = 1024, **kwargs):
    transform = Compose([
        Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(input_resolution),
        lambda x: x.convert('RGB'),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    return CaformerBackbone(input_resolution, heads, feat_dims, **kwargs), transform
