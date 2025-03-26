import os.path

import pytest
from hbutils.testing import tmatrix

from imgutils.data import pad_image_to_size, load_image, grid_transparent
from test.testings import get_testfile


@pytest.mark.unittest
class TestDataPad:
    @pytest.mark.parametrize(*tmatrix({
        'filename': [
            'nian_640_L.png',
            'nian_640_LA.png',
            'nian_640_RGB.png',
            'nian_640_RGBA.png',
            'png_640_m90.png',
            'png_640.png',
            'dori_640.png',
        ],
        'size': [
            (384, 512),
            (512, 384),
            (512, 512),
            (768, 512),
            (512, 768),
        ],
        'background_color': [
            'white',
            'black',
            'gray',
            'red',
            'green',
            'blue',
        ]
    }))
    def test_pad_image_to_size(self, filename, size, background_color, image_diff):
        actual_image = pad_image_to_size(
            get_testfile(filename),
            size=size,
            background_color=background_color
        )
        body, ext = os.path.splitext(filename)
        expected_image_file = get_testfile(f'{body}_{size[0]}x{size[1]}_{background_color}.png')
        expected_image = load_image(expected_image_file, mode=None, force_background=None)
        assert image_diff(
            grid_transparent(actual_image),
            grid_transparent(expected_image),
            throw_exception=False
        ) < 1e-2

    @pytest.mark.parametrize(*tmatrix({
        'filename': [
            'nian_640_L.png',
            'nian_640_LA.png',
            'nian_640_RGB.png',
            'nian_640_RGBA.png',
            'png_640_m90.png',
            'png_640.png',
            'dori_640.png',
        ],
        'size': [384, 512, 768],
        'background_color': [
            'white',
            'black',
            'gray',
            'red',
            'green',
            'blue',
        ]
    }))
    def test_pad_image_to_size_int_size(self, filename, size, background_color, image_diff):
        actual_image = pad_image_to_size(
            get_testfile(filename),
            size=size,
            background_color=background_color
        )
        body, ext = os.path.splitext(filename)
        expected_image_file = get_testfile(f'{body}_s{size}_{background_color}.png')
        expected_image = load_image(expected_image_file, mode=None, force_background=None)
        assert image_diff(
            grid_transparent(actual_image),
            grid_transparent(expected_image),
            throw_exception=False
        ) < 1e-2

    @pytest.mark.parametrize(*tmatrix({
        'filename': [
            'nian_640_L.png',
            'nian_640_LA.png',
            'nian_640_RGB.png',
            'nian_640_RGBA.png',
            'png_640_m90.png',
            'png_640.png',
            'dori_640.png',
        ],
        'size': [
            (384, 512),
            (512, 384),
            (512, 512),
            (768, 512),
            (512, 768),
        ],
        'background_color': [
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0, 128),
            (0, 255, 0, 128),
            (0, 0, 255, 128),
        ]
    }))
    def test_pad_image_to_tpl_color(self, filename, size, background_color, image_diff):
        actual_image = pad_image_to_size(
            get_testfile(filename),
            size=size,
            background_color=background_color
        )
        body, ext = os.path.splitext(filename)
        expected_image_file = get_testfile(f'{body}_{size[0]}x{size[1]}_{",".join(map(str, background_color))}.png')
        expected_image = load_image(expected_image_file, mode=None, force_background=None)
        assert image_diff(
            grid_transparent(actual_image),
            grid_transparent(expected_image),
            throw_exception=False
        ) < 1e-2

    @pytest.mark.parametrize(*tmatrix({
        'filename': [
            'nian_640_L.png',
            'nian_640_LA.png',
            'nian_640_RGB.png',
            'nian_640_RGBA.png',
            'png_640_m90.png',
            'png_640.png',
            'dori_640.png',
        ],
        'size': [
            (384, 512),
            (512, 384),
            (512, 512),
            (768, 512),
            (512, 768),
        ],
        'background_color': [
            0, 128, 255
        ]
    }))
    def test_pad_image_to_int_color(self, filename, size, background_color, image_diff):
        actual_image = pad_image_to_size(
            get_testfile(filename),
            size=size,
            background_color=background_color
        )
        body, ext = os.path.splitext(filename)
        expected_image_file = get_testfile(f'{body}_{size[0]}x{size[1]}_{background_color}.png')
        expected_image = load_image(expected_image_file, mode=None, force_background=None)
        assert image_diff(
            grid_transparent(actual_image),
            grid_transparent(expected_image),
            throw_exception=False
        ) < 1e-2

    def test_pad_image_to_size_invalid_size(self):
        with pytest.raises(TypeError):
            pad_image_to_size(
                get_testfile('nian.png'),
                size=(123, 234, 456),
                background_color='white'
            )

    def test_pad_image_to_size_invalid_color(self):
        with pytest.raises(TypeError):
            pad_image_to_size(
                get_testfile('nian.png'),
                size=(123, 234),
                background_color=object(),
            )

    def test_pad_image_to_size_invalid_image_mode(self):
        with pytest.raises(ValueError):
            pad_image_to_size(
                load_image(get_testfile('nian.png'), mode='RGB', force_background='white').convert('P'),
                size=(123, 234),
                background_color='white',
            )
