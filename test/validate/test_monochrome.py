import glob
import os.path
import random

import pytest
from hbutils.random import keep_global_state, global_seed
from hbutils.testing import tmatrix

from imgutils.validate.monochrome import get_monochrome_score, is_monochrome

_KNOWN_DUPS = {'2475192.jpg', '3842254.jpg', '2108110.jpg', '5257139.jpg', '6032011.jpg', '75719.jpg'}


@keep_global_state()
def get_samples():
    global_seed(0)
    all_samples_from_dataset = glob.glob(
        os.path.join('test', 'testfile', 'dataset', 'monochrome_danbooru', '*', '*.jpg'))
    files = random.sample(all_samples_from_dataset, k=30)
    return sorted([
        (os.path.basename(os.path.dirname(file)), os.path.basename(file))
        for file in files if os.path.basename(file) not in _KNOWN_DUPS
    ])


@pytest.mark.unittest
class TestValidateMonochrome:
    @pytest.mark.parametrize(*tmatrix({
        ('type_', 'file'): get_samples(),
        'safe': [0, 2, 4],
    }))
    def test_monochrome_test(self, type_: str, file: str, safe: int):
        filename = os.path.join('test', 'testfile', 'dataset', 'monochrome_danbooru', type_, file)
        if type_ == 'monochrome':
            assert get_monochrome_score(filename, safe=safe) >= 0.5
            assert is_monochrome(filename, safe=safe)
        else:
            assert get_monochrome_score(filename, safe=safe) <= 0.5
            assert not is_monochrome(filename, safe=safe)

    def test_monochrome_test_with_unknown_safe(self):
        with pytest.raises(ValueError):
            _ = get_monochrome_score(
                os.path.join('test', 'testfile', 'dataset', 'monochrome_danbooru', 'normal', '2475192.jpg'),
                safe=100
            )
        with pytest.raises(ValueError):
            _ = is_monochrome(
                os.path.join('test', 'testfile', 'dataset', 'monochrome_danbooru', 'normal', '2475192.jpg'),
                safe=100
            )
