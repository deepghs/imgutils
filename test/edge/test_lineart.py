import os.path
from unittest import skipUnless

import pytest
from hbutils.testing import tmatrix, OS

from imgutils.data import load_image
from imgutils.edge import edge_image_with_lineart
from imgutils.edge.lineart import _open_la_model
from test.testings import get_testfile


@pytest.fixture()
def _release_model_after_run():
    try:
        yield
    finally:
        _open_la_model.cache_clear()


@pytest.mark.unittest
class TestEdgeLineart:
    @skipUnless(OS.linux or OS.macos, 'Not run on windows')
    @pytest.mark.parametrize(*tmatrix({
        'original_image': ['6125785.jpg', '6125901.jpg'],
        'backcolor': ['transparent', 'white'],
        'forecolor': ['', 'black'],
        'coarse': [True, False],
    }))
    def test_edge_image_with_lineart(self, original_image, backcolor, forecolor, coarse,
                                     image_diff, _release_model_after_run):
        image = edge_image_with_lineart(
            get_testfile(original_image), coarse=coarse,
            backcolor=backcolor, forecolor=None if not forecolor else forecolor,
        )
        body, _ = os.path.splitext(original_image)

        assert image_diff(
            load_image(get_testfile(f'lineart_{body}_{backcolor}_{forecolor}_{coarse}.png'),
                       mode='RGB', force_background='pink'),
            load_image(image, mode='RGB', force_background='pink'),
            throw_exception=False
        ) < 1e-2
