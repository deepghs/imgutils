import os.path

import pytest
from hbutils.testing import tmatrix

from imgutils.validate.monochrome import get_monochrome_score, is_monochrome, _monochrome_validate_model


@pytest.fixture(scope='module', autouse=True)
def _release_model_after_run():
    try:
        yield
    finally:
        _monochrome_validate_model.cache_clear()


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
        ('model', 'safe'): [
            ('caformer_s36', False),
            ('mobilenetv3', False),
            ('mobilenetv3', True),
        ],
    }))
    def test_monochrome_test(self, type_: str, file: str, model: str, safe: bool):
        filename = os.path.join('test', 'testfile', 'dataset', 'monochrome_danbooru', type_, file)
        if type_ == 'monochrome':
            assert get_monochrome_score(filename, model=model, safe=safe) >= 0.5
            assert is_monochrome(filename, model=model, safe=safe)
        else:
            assert get_monochrome_score(filename, model=model, safe=safe) <= 0.5
            assert not is_monochrome(filename, model=model, safe=safe)

    def test_monochrome_test_with_unknown_safe(self):
        with pytest.raises(ValueError):
            _ = get_monochrome_score(
                os.path.join('test', 'testfile', 'dataset', 'monochrome_danbooru', 'normal', '2475192.jpg'),
                model='Model not found',
            )
