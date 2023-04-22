import glob
import os.path
import random

import pytest
from hbutils.random import keep_global_state, global_seed

from imgutils.validate.monochrome import get_monochrome_score, is_monochrome


@keep_global_state()
def get_samples():
    global_seed(0)
    all_samples_from_dataset = glob.glob(
        os.path.join('test', 'testfile', 'dataset', 'monochrome_danbooru', '*', '*.jpg'))
    files = random.sample(all_samples_from_dataset, k=30)
    return sorted([(os.path.basename(os.path.dirname(file)), os.path.basename(file)) for file in files])


@pytest.mark.unittest
class TestValidateMonochrome:
    @pytest.mark.parametrize(['type_', 'file'], get_samples())
    def test_monochrome_test(self, type_: str, file: str):
        filename = os.path.join('test', 'testfile', 'dataset', 'monochrome_danbooru', type_, file)
        if type_ == 'monochrome':
            assert get_monochrome_score(filename) >= 0.5
            assert is_monochrome(filename)
        else:
            assert get_monochrome_score(filename) <= 0.5
            assert not is_monochrome(filename)
