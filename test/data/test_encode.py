import pickle

import numpy as np
import pytest
from PIL import Image
from hbutils.testing import tmatrix

from imgutils.data import rgb_encode
from test.testings import get_testfile


@pytest.mark.unittest
class TestDataEncode:
    @pytest.mark.parametrize(*tmatrix({
        'filename': ['6124220.jpg', '6125785.png', '6125901.jpg'],
        'order_': ['CHW', 'CWH', 'WHC', 'HWC'],
        'use_float': [True, False]
    }))
    def test_rgb_encode(self, filename, order_, use_float):
        image = Image.open(get_testfile(filename))
        actual = rgb_encode(image, order_, use_float)

        with open(get_testfile(f'rgb_encode_{filename}_{order_}_{use_float}.pkl'), 'rb') as f:
            expected = pickle.load(f)

        assert np.isclose(actual, expected).all()
