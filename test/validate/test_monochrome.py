import os.path

import pytest
from hbutils.testing import tmatrix

from imgutils.generic.classify import _open_models_for_repo_id
from imgutils.validate.monochrome import get_monochrome_score, is_monochrome, _REPO_ID

_MODEL_NAMES = _open_models_for_repo_id(_REPO_ID).model_names


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _open_models_for_repo_id(_REPO_ID).clear()


def get_samples():
    return [
        ('monochrome', '6130053.jpg'),
        ('monochrome', '6125854（第 3 个复件）.jpg'),
        ('monochrome', '5221834.jpg'),
        ('monochrome', '1951253.jpg'),
        ('monochrome', '4879658.jpg'),
        ('monochrome', '80750471_p3_master1200.jpg'),

        ('normal', '54566940_p0_master1200.jpg'),
        ('normal', '60817155_p18_master1200.jpg'),
        ('normal', '4945494.jpg'),
        ('normal', '4008375.jpg'),
        ('normal', '2416278.jpg'),
        ('normal', '842709.jpg')
    ]


@pytest.mark.unittest
class TestValidateMonochrome:
    @pytest.mark.parametrize(*tmatrix({
        ('type_', 'file'): get_samples(),
        'model_name': _MODEL_NAMES,
    }))
    def test_monochrome_test(self, type_: str, file: str, model_name: str):
        filename = os.path.join('test', 'testfile', 'dataset', 'monochrome_danbooru', type_, file)
        if type_ == 'monochrome':
            assert get_monochrome_score(filename, model_name=model_name) >= 0.5
            assert is_monochrome(filename, model_name=model_name)
        else:
            assert get_monochrome_score(filename, model_name=model_name) <= 0.5
            assert not is_monochrome(filename, model_name=model_name)
