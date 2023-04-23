import os.path

import pytest
from hbutils.testing import tmatrix

from imgutils.validate.monochrome import get_monochrome_score, is_monochrome


def get_samples():
    return [
        ('monochrome', '143640.jpg'), ('monochrome', '2165075.jpg'), ('monochrome', '2267010.jpg'),
        ('monochrome', '2642558.jpg'), ('monochrome', '3141176.jpg'), ('monochrome', '4530291.jpg'),
        ('monochrome', '4589191.jpg'), ('monochrome', '5182260.jpg'), ('monochrome', '5376761.jpg'),
        ('monochrome', '5608827.jpg'), ('monochrome', '5992785.jpg'), ('monochrome', '6126963.jpg'),
        ('monochrome', '6128358.jpg'), ('monochrome', '6131733.jpg'), ('monochrome', '6154723.jpg'),
        ('monochrome', '843419.jpg'), ('monochrome', '84584446_p3_master1200.jpg'),
        ('monochrome', '87392919_p26_master1200.jpg'),

        ('normal', '2034501.jpg'), ('normal', '2160617.jpg'), ('normal', '3446505.jpg'),
        ('normal', '3725624.jpg'), ('normal', '3899045.jpg'), ('normal', '4278075.jpg'),
        ('normal', '4897680.jpg'), ('normal', '5531563.jpg'), ('normal', '62722650_p14_master1200.jpg'),
        ('normal', '86243980_p0_master1200.jpg'), ('normal', '89270548_p3_master1200.jpg')
    ]


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
