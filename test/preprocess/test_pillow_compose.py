import textwrap
from unittest import skipUnless

import numpy as np
import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.preprocess.pillow import PillowNormalize, PillowCompose, PillowResize, PillowMaybeToTensor, \
    PillowCenterCrop
from test.testings import get_testfile

try:
    import torchvision
except (ImportError, ModuleNotFoundError):
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


@pytest.fixture()
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


@pytest.fixture()
def torchvision_mobilenet(torchvision_maybetotensor):
    import torch
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
    return Compose([
        Resize(size=404, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        CenterCrop(size=[384, 384]),
        torchvision_maybetotensor,
        Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
    ])


@pytest.fixture()
def pillow_mobilenet():
    return PillowCompose([
        PillowResize(size=404, interpolation=Image.BICUBIC, max_size=None, antialias=True),
        PillowCenterCrop(size=(384, 384)),
        PillowMaybeToTensor(),
        PillowNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@pytest.fixture()
def torchvision_beit(torchvision_maybetotensor):
    import torch
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
    return Compose([
        Resize(size=384, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        CenterCrop(size=[384, 384]),
        torchvision_maybetotensor,
        Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]), std=torch.tensor([0.5000, 0.5000, 0.5000])),
    ])


@pytest.fixture()
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
    def test_compose_mobilenet(self, src_image, pillow_mobilenet, torchvision_mobilenet):
        image = Image.open(get_testfile(src_image))
        presult = pillow_mobilenet(image)
        tresult = torchvision_mobilenet(image)
        np.testing.assert_array_almost_equal(presult, tresult.numpy())

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
    }))
    def test_compose_beit(self, src_image, pillow_beit, torchvision_beit):
        image = Image.open(get_testfile(src_image))
        presult = pillow_beit(image)
        tresult = torchvision_beit(image)
        np.testing.assert_array_almost_equal(presult, tresult.numpy())

    def test_compose_repr(self, pillow_mobilenet, pillow_beit):
        assert textwrap.dedent(repr(pillow_mobilenet)).strip() == \
               textwrap.dedent("""
PillowCompose(
    PillowResize(size=404, interpolation=bicubic, max_size=None, antialias=True)
    PillowCenterCrop(size=(384, 384))
    PillowMaybeToTensor()
    PillowNormalize(mean=[0.485 0.456 0.406], std=[0.229 0.224 0.225])
)
            """).strip()

        assert textwrap.dedent(repr(pillow_beit)).strip() == \
               textwrap.dedent("""
PillowCompose(
    PillowResize(size=384, interpolation=bicubic, max_size=None, antialias=True)
    PillowCenterCrop(size=(384, 384))
    PillowMaybeToTensor()
    PillowNormalize(mean=[0.5 0.5 0.5], std=[0.5 0.5 0.5])
)
            """).strip()
