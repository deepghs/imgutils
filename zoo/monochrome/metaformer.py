import torch
from torch import nn
from timm.models import create_model
import zoo.monochrome.metaformer_timm # register models

class CAFormerBuilder:
    __model_name__ = 'caformer'
    def __init__(self, arch='caformer_m36_384_in21ft1k', num_classes=2, drop_path_rate=0.4):
        self.create_model_args = dict(
            model_name=arch,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.0,
            drop_connect_rate=None,  # DEPRECATED, use drop_path
            drop_path_rate=drop_path_rate,
            drop_block_rate=None,
            global_pool=None,
            bn_momentum=None,
            bn_eps=None,
            scriptable=False,
            checkpoint_path=None
        )
        self.num_classes=num_classes

    def __call__(self, *args, **kwargs):
        model = create_model(**self.create_model_args)
        return model

if __name__ == '__main__':
    from thop import profile

    transformer = CAFormerBuilder()()
    x = torch.randn(1, 3, 384, 384)

    flops, params = profile(transformer, (x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')