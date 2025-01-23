import textwrap
from unittest import skipUnless

import numpy as np
import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.preprocess.pillow import PillowNormalize, PillowCompose, PillowResize, PillowMaybeToTensor, \
    PillowCenterCrop, create_pillow_transforms
from imgutils.preprocess.torchvision import create_torchvision_transforms
from test.testings import get_testfile

try:
    import torchvision
except (ImportError, ModuleNotFoundError):
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


@pytest.fixture()
def mobilenetv4_conv_large_e600_r384_in1k():
    return [{'antialias': True,
             'interpolation': 'bicubic',
             'max_size': None,
             'size': 404,
             'type': 'resize'},
            {'size': [384, 384], 'type': 'center_crop'},
            {'type': 'maybe_to_tensor'},
            {'mean': [0.48500001430511475, 0.4560000002384186, 0.4059999883174896],
             'std': [0.2290000021457672, 0.2240000069141388, 0.22499999403953552],
             'type': 'normalize'}]


@pytest.fixture()
def caformer_s36_sail_in1k_384():
    return [{'antialias': True,
             'interpolation': 'bicubic',
             'max_size': None,
             'size': 384,
             'type': 'resize'},
            {'size': [384, 384], 'type': 'center_crop'},
            {'type': 'maybe_to_tensor'},
            {'mean': [0.48500001430511475, 0.4560000002384186, 0.4059999883174896],
             'std': [0.2290000021457672, 0.2240000069141388, 0.22499999403953552],
             'type': 'normalize'}]


@pytest.fixture()
def beit_base_patch16_384_in22k_ft_in22k_in1k():
    return [{'antialias': True,
             'interpolation': 'bicubic',
             'max_size': None,
             'size': 384,
             'type': 'resize'},
            {'size': [384, 384], 'type': 'center_crop'},
            {'type': 'maybe_to_tensor'},
            {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'type': 'normalize'}]


@pytest.fixture()
def resnet101d_ra2_in1k():
    return [{'antialias': True,
             'interpolation': 'bicubic',
             'max_size': None,
             'size': 269,
             'type': 'resize'},
            {'size': [256, 256], 'type': 'center_crop'},
            {'type': 'maybe_to_tensor'},
            {'mean': [0.48500001430511475, 0.4560000002384186, 0.4059999883174896],
             'std': [0.2290000021457672, 0.2240000069141388, 0.22499999403953552],
             'type': 'normalize'}]


@pytest.fixture()
def meta_collect(mobilenetv4_conv_large_e600_r384_in1k, resnet101d_ra2_in1k, caformer_s36_sail_in1k_384,
                 beit_base_patch16_384_in22k_ft_in22k_in1k):
    return {
        'mobilenetv4_conv_large.e600_r384_in1k': mobilenetv4_conv_large_e600_r384_in1k,
        'resnet101d.ra2_in1k': resnet101d_ra2_in1k,
        'caformer_s36.sail_in1k_384': caformer_s36_sail_in1k_384,
        'beit_base_patch16_384.in22k_ft_in22k_in1k': beit_base_patch16_384_in22k_ft_in22k_in1k,
    }


def torchvision_maybetotensor():
    from torchvision.transforms import ToTensor
    class MaybeToTensor(ToTensor):
        def __init__(self) -> None:
            super().__init__()

        def __call__(self, pic):
            import torchvision.transforms.functional as F
            import torch
            if isinstance(pic, torch.Tensor):
                return pic
            return F.to_tensor(pic)

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}()"

    return MaybeToTensor()


def torchvision_mobilenet():
    import torch
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
    return Compose([
        Resize(size=404, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        CenterCrop(size=[384, 384]),
        torchvision_maybetotensor(),
        Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
    ])


def pillow_mobilenet():
    return PillowCompose([
        PillowResize(size=404, interpolation=Image.BICUBIC, max_size=None, antialias=True),
        PillowCenterCrop(size=(384, 384)),
        PillowMaybeToTensor(),
        PillowNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def torchvision_beit():
    import torch
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
    return Compose([
        Resize(size=384, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        CenterCrop(size=[384, 384]),
        torchvision_maybetotensor(),
        Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]), std=torch.tensor([0.5000, 0.5000, 0.5000])),
    ])


def pillow_beit():
    return PillowCompose([
        PillowResize(size=384, interpolation=Image.BICUBIC, max_size=None, antialias=True),
        PillowCenterCrop(size=(384, 384)),
        PillowMaybeToTensor(),
        PillowNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


@pytest.mark.unittest
class TestPreprocessPillowCompose:
    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
    }))
    def test_compose_mobilenet(self, src_image):
        image = Image.open(get_testfile(src_image))
        presult = pillow_mobilenet()(image)
        tresult = torchvision_mobilenet()(image)
        np.testing.assert_array_almost_equal(presult, tresult.numpy())

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
    }))
    def test_compose_beit(self, src_image):
        image = Image.open(get_testfile(src_image))
        presult = pillow_beit()(image)
        tresult = torchvision_beit()(image)
        np.testing.assert_array_almost_equal(presult, tresult.numpy())

    def test_compose_repr(self):
        assert textwrap.dedent(repr(pillow_mobilenet())).strip() == \
               textwrap.dedent("""
PillowCompose(
    PillowResize(size=404, interpolation=bicubic, max_size=None, antialias=True)
    PillowCenterCrop(size=(384, 384))
    PillowMaybeToTensor()
    PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229 0.224 0.225])
)
            """).strip()

        assert textwrap.dedent(repr(pillow_beit())).strip() == \
               textwrap.dedent("""
PillowCompose(
    PillowResize(size=384, interpolation=bicubic, max_size=None, antialias=True)
    PillowCenterCrop(size=(384, 384))
    PillowMaybeToTensor()
    PillowNormalize(mean=[0.5 0.5 0.5], std=[0.5 0.5 0.5])
)
            """).strip()

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        'meta_name': [
            'mobilenetv4_conv_large.e600_r384_in1k',
            'caformer_s36.sail_in1k_384',
            'beit_base_patch16_384.in22k_ft_in22k_in1k',
            'resnet101d.ra2_in1k',
        ]
    }))
    def test_compose_alignment(self, src_image, meta_name, meta_collect):
        image = Image.open(get_testfile(src_image))
        meta = meta_collect[meta_name]

        ptrans = create_pillow_transforms(meta)
        ttrans = create_torchvision_transforms(meta)
        np.testing.assert_array_almost_equal(
            ptrans(image),
            ttrans(image).numpy(),
        )


    def test_create_transform_invalid(self):
        with pytest.raises(TypeError):
            _ = create_pillow_transforms(None)
        with pytest.raises(TypeError):
            _ = create_pillow_transforms(1)
        with pytest.raises(TypeError):
            _ = create_pillow_transforms('str')

