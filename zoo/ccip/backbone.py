from functools import partial
from typing import Tuple, Dict, Callable

import clip
import torch
from torchvision.transforms import Compose

from .caformer import get_caformer, get_caformer_s18, get_caformer_query, get_caformer_b36, get_caformer_s36


def get_clip_backbone(name="ViT-B/32") -> Tuple[torch.nn.Module, Compose]:
    model, preprocess = clip.load(name, device='cpu')
    return model.visual.type(torch.float32), preprocess


CLIP_PREFIX = 'clip/'
_KNOWN_BACKBONES: Dict[str, Callable[..., Tuple[torch.nn.Module, Compose]]] = {}


def register_backbone(name, func, *args, **kwargs):
    _KNOWN_BACKBONES[name] = partial(func, *args, **kwargs)


register_backbone('caformer', get_caformer)
register_backbone('caformer_query', get_caformer_query)
register_backbone('caformer_s18', get_caformer_s18)
register_backbone('caformer_s36', get_caformer_s36)
register_backbone('caformer_b36', get_caformer_b36)


def get_backbone(name: str) -> Tuple[torch.nn.Module, Compose]:
    if name.startswith(CLIP_PREFIX):
        clip_name = name[len(CLIP_PREFIX):]
        if clip_name in clip.available_models():
            return get_clip_backbone(clip_name)
        else:
            raise ValueError(f'Unknown model in clip - {clip_name!r}.')
    else:
        if name in _KNOWN_BACKBONES:
            return _KNOWN_BACKBONES[name]()
        else:
            raise ValueError(f'Unknown backbone - {name!r}.')
