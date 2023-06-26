import torch.nn
from torchvision.transforms import Normalize

from .attention_pool import AttentionPool2d, AttentionPool2d_query, AttentionPool2d_flat
from ..monochrome.metaformer import CAFormerBuilder
from torch import nn

class CaformerBackbone(torch.nn.Module):
    def __init__(self, input_resolution: int = 384, heads: int = 8, out_dims: int = 768, pool_with_query=False, **kwargs):
        torch.nn.Module.__init__(self)
        self.input_resolution = input_resolution
        self.caformer = CAFormerBuilder(**kwargs)()
        if pool_with_query:
            self.attnpool = nn.Sequential(
                AttentionPool2d_query(self.input_resolution // 32, self.caformer.output_dim, heads, self.caformer.output_dim, n_query=8),
                AttentionPool2d_flat(8, self.caformer.output_dim, heads, out_dims),
            )
        else:
            self.attnpool = AttentionPool2d(self.input_resolution//32, self.caformer.output_dim, heads, out_dims)

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


def get_caformer(input_resolution: int = 384, heads: int = 8, feat_dims: int = 768, **kwargs):
    transform = [
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]

    return CaformerBackbone(input_resolution, heads, feat_dims, **kwargs), transform

def get_caformer_query(input_resolution: int = 384, heads: int = 8, feat_dims: int = 768, **kwargs):
    transform = [
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]

    return CaformerBackbone(input_resolution, heads, feat_dims, pool_with_query=True, **kwargs), transform

def get_caformer_s18(input_resolution: int = 384, heads: int = 8, feat_dims: int = 768, **kwargs):
    transform = [
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]

    return CaformerBackbone(input_resolution, heads, feat_dims, arch='caformer_s18_384_in21ft1k', **kwargs), transform