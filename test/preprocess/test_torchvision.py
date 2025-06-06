from typing import Union, Tuple
from unittest import skipUnless

import numpy as np
import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.data import load_image, grid_transparent
from imgutils.preprocess import NotParseTarget, create_pillow_transforms
from imgutils.preprocess.torchvision import _get_interpolation_mode, create_torchvision_transforms, \
    parse_torchvision_transforms, register_torchvision_transform, register_torchvision_parse, PadToSize
from test.testings import get_testfile

try:
    import torch
    import torchvision
except (ImportError, ModuleNotFoundError):
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


@pytest.mark.unittest
class TestPreprocessPillow:
    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_get_interpolation_mode_raw(self):
        from torchvision.transforms import InterpolationMode
        assert _get_interpolation_mode(InterpolationMode.NEAREST) == InterpolationMode.NEAREST
        assert _get_interpolation_mode(InterpolationMode.LANCZOS) == InterpolationMode.LANCZOS
        assert _get_interpolation_mode(InterpolationMode.BILINEAR) == InterpolationMode.BILINEAR
        assert _get_interpolation_mode(InterpolationMode.BICUBIC) == InterpolationMode.BICUBIC
        assert _get_interpolation_mode(InterpolationMode.BOX) == InterpolationMode.BOX
        assert _get_interpolation_mode(InterpolationMode.HAMMING) == InterpolationMode.HAMMING

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_get_interpolation_mode_int(self):
        from torchvision.transforms import InterpolationMode
        assert _get_interpolation_mode(0) == InterpolationMode.NEAREST
        assert _get_interpolation_mode(1) == InterpolationMode.LANCZOS
        assert _get_interpolation_mode(2) == InterpolationMode.BILINEAR
        assert _get_interpolation_mode(3) == InterpolationMode.BICUBIC
        assert _get_interpolation_mode(4) == InterpolationMode.BOX
        assert _get_interpolation_mode(5) == InterpolationMode.HAMMING
        with pytest.raises(ValueError):
            _get_interpolation_mode(-1)
        with pytest.raises(ValueError):
            _get_interpolation_mode(100)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_get_interpolation_mode_str(self):
        from torchvision.transforms import InterpolationMode
        assert _get_interpolation_mode('nearest') == InterpolationMode.NEAREST
        assert _get_interpolation_mode('NEAREST') == InterpolationMode.NEAREST
        assert _get_interpolation_mode('bilinear') == InterpolationMode.BILINEAR
        assert _get_interpolation_mode('bicubic') == InterpolationMode.BICUBIC
        assert _get_interpolation_mode('box') == InterpolationMode.BOX
        assert _get_interpolation_mode('hamming') == InterpolationMode.HAMMING
        assert _get_interpolation_mode('lanczos') == InterpolationMode.LANCZOS
        with pytest.raises(ValueError):
            _ = _get_interpolation_mode('xxx')
        with pytest.raises(ValueError):
            _ = _get_interpolation_mode('')

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_get_interpolation_mode_invalid(self):
        with pytest.raises(TypeError):
            _ = _get_interpolation_mode(None)
        with pytest.raises(TypeError):
            _ = _get_interpolation_mode([])

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        ('mode', 'channels'): [
            ('I', 1), ('I;16', 1), ('F', 1), ('1', 1), ('L', 1), ('P', 1),
            ('LA', 2),
            ('RGB', 3), ('YCbCr', 3),
            ('RGBA', 4), ('CMYK', 4),
        ]
    }))
    def test_maybe_to_tensor(self, src_image, mode, channels):
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        ttrans = create_torchvision_transforms({'type': 'maybe_to_tensor'})
        result = ttrans(image)
        assert tuple(result.shape) == (channels, image.height, image.width)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_maybe_to_tensor_repr(self):
        ttrans = create_torchvision_transforms({'type': 'maybe_to_tensor'})
        assert repr(ttrans) == 'MaybeToTensor()'

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_maybe_to_tensor_np(self):
        import torch
        input_ = torch.randn(3, 384, 384)
        ttrans = create_torchvision_transforms({'type': 'maybe_to_tensor'})
        assert torch.allclose(ttrans(input_), input_)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'src_image': [
            'png_640.png',
            'png_640_m90.png',
        ],
        ('mode', 'channels'): [
            ('I', 1), ('I;16', 1), ('F', 1), ('1', 1), ('L', 1), ('P', 1),
            ('LA', 2),
            ('RGB', 3), ('YCbCr', 3),
            ('RGBA', 4), ('CMYK', 4),
        ]
    }))
    def test_to_tensor(self, src_image, mode, channels):
        image = Image.open(get_testfile(src_image))
        image = image.convert(mode)
        assert image.mode == mode

        ttrans = create_torchvision_transforms({'type': 'to_tensor'})
        result = ttrans(image)
        assert tuple(result.shape) == (channels, image.height, image.width)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_create_transform_invalid(self):
        with pytest.raises(TypeError):
            _ = create_torchvision_transforms(None)
        with pytest.raises(TypeError):
            _ = create_torchvision_transforms(1)
        with pytest.raises(TypeError):
            _ = create_torchvision_transforms('str')

    @skipUnless(not _TORCHVISION_AVAILABLE, 'Non-torchvision required.')
    def test_create_transform_non_torchvision(self):
        with pytest.raises(EnvironmentError):
            _ = create_torchvision_transforms([])

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required')
    def test_parse_torchvision_transforms(self):
        import torch
        from torchvision.transforms import Compose, Resize, InterpolationMode, CenterCrop, Normalize, ToTensor

        assert parse_torchvision_transforms(Compose([
            Resize(size=384, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            CenterCrop(size=[384, 384]),
            create_torchvision_transforms({'type': 'maybe_to_tensor'}),
            Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]), std=torch.tensor([0.5000, 0.5000, 0.5000])),
        ])) == [
                   {'antialias': True,
                    'interpolation': 'bicubic',
                    'max_size': None,
                    'size': 384,
                    'type': 'resize'},
                   {'size': [384, 384], 'type': 'center_crop'},
                   {'type': 'maybe_to_tensor'},
                   {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'type': 'normalize'}
               ]

        assert parse_torchvision_transforms(Compose([
            Resize(size=384, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            CenterCrop(size=[384, 384]),
            ToTensor(),
            Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]), std=torch.tensor([0.5000, 0.5000, 0.5000])),
        ])) == [
                   {'antialias': True,
                    'interpolation': 'bicubic',
                    'max_size': None,
                    'size': 384,
                    'type': 'resize'},
                   {'size': [384, 384], 'type': 'center_crop'},
                   {'type': 'to_tensor'},
                   {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'type': 'normalize'}
               ]

        assert parse_torchvision_transforms(
            Resize(size=384, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)) \
               == {'antialias': True,
                   'interpolation': 'bicubic',
                   'max_size': None,
                   'size': 384,
                   'type': 'resize'}
        assert parse_torchvision_transforms(CenterCrop(size=[384, 384])) == {'size': [384, 384], 'type': 'center_crop'}
        assert parse_torchvision_transforms(ToTensor()) == {'type': 'to_tensor'}
        assert parse_torchvision_transforms(
            Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]), std=torch.tensor([0.5000, 0.5000, 0.5000]))) \
               == {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'type': 'normalize'}

        with pytest.raises(TypeError):
            _ = parse_torchvision_transforms(None)
        with pytest.raises(TypeError):
            _ = parse_torchvision_transforms(23344)

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_register_and_use(self):
        from torchvision.transforms import ColorJitter
        @register_torchvision_transform('color_jitter')
        def _create_color_jitter(
                brightness: Union[float, Tuple[float, float]] = 0,
                contrast: Union[float, Tuple[float, float]] = 0,
                saturation: Union[float, Tuple[float, float]] = 0,
                hue: Union[float, Tuple[float, float]] = 0,
        ):
            return ColorJitter(brightness, contrast, saturation, hue)

        @register_torchvision_parse('color_jitter')
        def _parse_color_jitter(obj: ColorJitter):
            if not isinstance(obj, ColorJitter):
                raise NotParseTarget

            return {
                'brightness': obj.brightness,
                'contrast': obj.contrast,
                'saturation': obj.saturation,
                'hue': obj.hue,
            }

        c = create_torchvision_transforms({
            'type': 'color_jitter',
            'brightness': 0.5,
            'contrast': 0.2,
            'saturation': (0.0, 0.8),
            'hue': (0.1, 0.45),
        })
        assert isinstance(c, ColorJitter)
        assert c.brightness == pytest.approx((0.5, 1.5))
        assert c.contrast == pytest.approx((0.8, 1.2))
        assert c.saturation == pytest.approx((0.0, 0.8))
        assert c.hue == pytest.approx((0.1, 0.45))

        assert parse_torchvision_transforms(c) == pytest.approx({
            'brightness': (0.5, 1.5),
            'contrast': (0.8, 1.2),
            'hue': (0.1, 0.45),
            'saturation': (0.0, 0.8),
            'type': 'color_jitter'
        })

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'json_': [
            {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'nearest'},
            {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'gray', 'interpolation': 'lanczos'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255), 'interpolation': 'bicubic'},
            {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'blue', 'interpolation': 'box'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 255, 255),
             'interpolation': 'bilinear'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'blue', 'interpolation': 'lanczos'},
            {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'nearest'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 0, 0, 128), 'interpolation': 'box'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255, 128),
             'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255, 128),
             'interpolation': 'nearest'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255), 'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 255, 255),
             'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 0, 0, 128),
             'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'white', 'interpolation': 'bilinear'},
        ],
        'filename': [
            'nian_640_L.png',
            'nian_640_LA.png',
            'nian_640_RGB.png',
            'nian_640_RGBA.png',
            'png_640_m90.png',
            'png_640.png',
            'dori_640.png',
        ]
    }, mode='matrix'))
    def test_align_pad_to_size(self, json_, filename, image_diff):
        src_image = load_image(get_testfile(filename), mode=None, force_background=None)
        tprocess = create_torchvision_transforms(json_)
        pprocess = create_pillow_transforms(json_)
        assert image_diff(
            grid_transparent(tprocess(src_image)),
            grid_transparent(pprocess(src_image)),
            throw_exception=False
        ) < 1e-3

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(['json_from', 'json_to'], [
        ({'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'nearest'},
         {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'nearest'}),
        ({'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'gray', 'interpolation': 'lanczos'},
         {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'gray', 'interpolation': 'lanczos'}),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255), 'interpolation': 'bicubic'},
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255], 'interpolation': 'bicubic'}),
        ({'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'blue', 'interpolation': 'box'},
         {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'blue', 'interpolation': 'box'}),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 255, 255), 'interpolation': 'bilinear'},
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 255, 255], 'interpolation': 'bilinear'}),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'blue', 'interpolation': 'lanczos'},
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'blue', 'interpolation': 'lanczos'}),
        ({'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'nearest'},
         {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'nearest'}),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'hamming'},
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'hamming'}),
        ({'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 0, 0, 128), 'interpolation': 'box'},
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 0, 0, 128], 'interpolation': 'box'}),
        ({'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255, 128), 'interpolation': 'hamming'},
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255, 128], 'interpolation': 'hamming'}),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255, 128), 'interpolation': 'nearest'},
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [0, 0, 255, 128], 'interpolation': 'nearest'}),
        ({'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255), 'interpolation': 'hamming'},
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [0, 0, 255], 'interpolation': 'hamming'}),
        ({'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 255, 255), 'interpolation': 'hamming'},
         {'type': 'pad_to_size', 'size': [384, 384], 'background_color': [255, 255, 255], 'interpolation': 'hamming'}),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 0, 0, 128), 'interpolation': 'hamming'},
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': [255, 0, 0, 128], 'interpolation': 'hamming'}),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'white', 'interpolation': 'bilinear'},
         {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'white', 'interpolation': 'bilinear'}),

    ])
    def test_pad_to_size_to_json(self, json_from, json_to):
        assert parse_torchvision_transforms(create_torchvision_transforms(json_from)) == json_to

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(['json_', 'repr_text'], [
        ({'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'nearest'},
         'PadToSize(size=(512, 768), interpolation=nearest, background_color=white)'),
        ({'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'gray', 'interpolation': 'lanczos'},
         'PadToSize(size=(768, 512), interpolation=lanczos, background_color=gray)'),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255), 'interpolation': 'bicubic'},
         'PadToSize(size=(512, 512), interpolation=bicubic, background_color=(0, 0, 255))'),
        ({'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'blue', 'interpolation': 'box'},
         'PadToSize(size=(384, 512), interpolation=box, background_color=blue)'),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 255, 255), 'interpolation': 'bilinear'},
         'PadToSize(size=(512, 512), interpolation=bilinear, background_color=(255, 255, 255))'),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'blue', 'interpolation': 'lanczos'},
         'PadToSize(size=(512, 512), interpolation=lanczos, background_color=blue)'),
        ({'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'nearest'},
         'PadToSize(size=(768, 512), interpolation=nearest, background_color=red)'),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'hamming'},
         'PadToSize(size=(512, 512), interpolation=hamming, background_color=gray)'),
        ({'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 0, 0, 128), 'interpolation': 'box'},
         'PadToSize(size=(384, 384), interpolation=box, background_color=(255, 0, 0, 128))'),
        ({'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255, 128), 'interpolation': 'hamming'},
         'PadToSize(size=(384, 384), interpolation=hamming, background_color=(0, 0, 255, 128))'),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255, 128), 'interpolation': 'nearest'},
         'PadToSize(size=(512, 512), interpolation=nearest, background_color=(0, 0, 255, 128))'),
        ({'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255), 'interpolation': 'hamming'},
         'PadToSize(size=(384, 384), interpolation=hamming, background_color=(0, 0, 255))'),
        ({'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 255, 255), 'interpolation': 'hamming'},
         'PadToSize(size=(384, 384), interpolation=hamming, background_color=(255, 255, 255))'),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 0, 0, 128), 'interpolation': 'hamming'},
         'PadToSize(size=(512, 512), interpolation=hamming, background_color=(255, 0, 0, 128))'),
        ({'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'white', 'interpolation': 'bilinear'},
         'PadToSize(size=(512, 512), interpolation=bilinear, background_color=white)'),
    ])
    def test_pad_to_size_repr_text(self, json_, repr_text):
        assert repr(create_torchvision_transforms(json_)) == repr_text

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_pad_to_size_error(self):
        with pytest.raises(TypeError):
            PadToSize((512, 512))(np.random.randn(1, 3, 384, 384))

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'json_': [
            # {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'nearest'},
            {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'gray', 'interpolation': 'lanczos'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255), 'interpolation': 'bicubic'},
            {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'blue', 'interpolation': 'box'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 255, 255),
             'interpolation': 'bilinear'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'blue', 'interpolation': 'lanczos'},
            # {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'nearest'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 0, 0, 128), 'interpolation': 'box'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255, 128),
             'interpolation': 'hamming'},
            # {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255, 128),
            #  'interpolation': 'nearest'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255), 'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 255, 255),
             'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 0, 0, 128),
             'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'white', 'interpolation': 'bilinear'},
        ],
        'filename': [
            'nian_640_L.png',
            'nian_640_LA.png',
            'nian_640_RGB.png',
            'nian_640_RGBA.png',
            'png_640_m90.png',
            'png_640.png',
            'dori_640.png',
        ]
    }, mode='matrix'))
    def test_pad_to_size_tensor_and_image_float(self, json_, filename, image_diff):
        import torch
        p = create_torchvision_transforms(json_)
        image = load_image(get_testfile(filename), force_background=None, mode=None)

        expected_image = p(image)
        if image.mode == 'L':
            tensor = torch.tensor(np.array(image)[np.newaxis, ...] / 255.0).type(torch.float32)
        else:
            tensor = torch.tensor(np.array(image).transpose(2, 0, 1) / 255.0).type(torch.float32)
        actual_tensor = p(tensor)
        if image.mode == 'L':
            actual_image = Image.fromarray(
                (actual_tensor.numpy()[0] * 255.0).astype(np.uint8),
                mode=image.mode
            )
        else:
            actual_image = Image.fromarray(
                (actual_tensor.numpy().transpose((1, 2, 0)) * 255.0).astype(np.uint8),
                mode=image.mode
            )

        assert tensor.dtype == actual_tensor.dtype
        assert tensor.device == actual_tensor.device

        assert image_diff(
            grid_transparent(expected_image),
            grid_transparent(actual_image),
            throw_exception=False,
        ) < 1.5e-2

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    @pytest.mark.parametrize(*tmatrix({
        'json_': [
            # {'type': 'pad_to_size', 'size': [512, 768], 'background_color': 'white', 'interpolation': 'nearest'},
            {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'gray', 'interpolation': 'lanczos'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255), 'interpolation': 'bicubic'},
            {'type': 'pad_to_size', 'size': [384, 512], 'background_color': 'blue', 'interpolation': 'box'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 255, 255),
             'interpolation': 'bilinear'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'blue', 'interpolation': 'lanczos'},
            # {'type': 'pad_to_size', 'size': [768, 512], 'background_color': 'red', 'interpolation': 'nearest'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'gray', 'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 0, 0, 128), 'interpolation': 'box'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255, 128),
             'interpolation': 'hamming'},
            # {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (0, 0, 255, 128),
            #  'interpolation': 'nearest'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (0, 0, 255), 'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [384, 384], 'background_color': (255, 255, 255),
             'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': (255, 0, 0, 128),
             'interpolation': 'hamming'},
            {'type': 'pad_to_size', 'size': [512, 512], 'background_color': 'white', 'interpolation': 'bilinear'},
        ],
        'filename': [
            'nian_640_L.png',
            'nian_640_LA.png',
            'nian_640_RGB.png',
            'nian_640_RGBA.png',
            'png_640_m90.png',
            'png_640.png',
            'dori_640.png',
        ]
    }, mode='matrix'))
    def test_pad_to_size_tensor_and_image_int8(self, json_, filename, image_diff):
        import torch
        p = create_torchvision_transforms(json_)
        image = load_image(get_testfile(filename), force_background=None, mode=None)

        expected_image = p(image)
        if image.mode == 'L':
            tensor = torch.tensor(np.array(image)[np.newaxis, ...]).type(torch.uint8)
        else:
            tensor = torch.tensor(np.array(image).transpose(2, 0, 1)).type(torch.uint8)
        actual_tensor = p(tensor)
        if image.mode == 'L':
            actual_image = Image.fromarray(
                (actual_tensor.numpy()[0]).astype(np.uint8),
                mode=image.mode
            )
        else:
            actual_image = Image.fromarray(
                (actual_tensor.numpy().transpose((1, 2, 0))).astype(np.uint8),
                mode=image.mode
            )

        assert tensor.dtype == actual_tensor.dtype
        assert tensor.device == actual_tensor.device

        assert image_diff(
            grid_transparent(expected_image),
            grid_transparent(actual_image),
            throw_exception=False,
        ) < 1.5e-2

    @skipUnless(_TORCHVISION_AVAILABLE, 'Torchvision required.')
    def test_pad_to_size_dim_error(self):
        import torch
        with pytest.raises(ValueError):
            PadToSize((512, 512))(torch.randn(384, 384))
        with pytest.raises(ValueError):
            PadToSize((512, 512))(torch.randn(1, 1, 1, 384, 384))
