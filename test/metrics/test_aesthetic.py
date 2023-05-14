import os.path

import pytest

from imgutils.metrics import get_aesthetic_score
from imgutils.metrics.aesthetic import _open_aesthetic_model
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model():
    try:
        yield
    finally:
        _open_aesthetic_model.cache_clear()


@pytest.mark.unittest
class TestMetricsAesthetic:
    @pytest.mark.parametrize(['file', 'score'], [
        ('2053756-0.100.jpg', 0.09986039996147156),
        ('1663584-0.243.jpg', 0.24299287796020508),
        ('4886411-0.381.jpg', 0.38091593980789185),
        ('2066024-0.513.jpg', 0.5131649971008301),
        ('3670169-0.601.jpg', 0.6011670827865601),
        ('5930006-0.707.jpg', 0.7067991495132446),
        ('3821265-0.824.jpg', 0.8237218260765076),
        ('5512471-0.919.jpg', 0.9187621474266052),
    ])
    def test_get_aesthetic_score(self, file, score):
        assert get_aesthetic_score(get_testfile(os.path.join('aesthetic', file))) == \
               pytest.approx(score, abs=1e-3)
