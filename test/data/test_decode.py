import pickle

import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.data import rgb_decode
from test.testings import get_testfile


@pytest.mark.unittest
class TestDataDecode:
    @pytest.mark.parametrize(*tmatrix({
        'filename': ['6124220.jpg', '6125785.png', '6125901.jpg'],
        'order_': ['CHW', 'CWH', 'WHC', 'HWC'],
        'use_float': [True, False]
    }))
    def test_rgb_decode(self, filename, order_, use_float, image_diff):
        with open(get_testfile(f'rgb_encode_{filename}_{order_}_{use_float}.pkl'), 'rb') as f:
            data = pickle.load(f)

        # print(data)

        actual = rgb_decode(data, order_=order_)
        expected = Image.open(get_testfile(filename)).convert('RGB')

        assert image_diff(actual, expected, throw_exception=False) < 1e-2
