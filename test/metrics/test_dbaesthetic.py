import pytest

from imgutils.metrics import anime_dbaesthetic
from imgutils.metrics.dbaesthetic import _MODEL, _LABELS
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model():
    try:
        yield
    finally:
        _MODEL.clear()


@pytest.mark.unittest
class TestMetricsDBAesthetic:
    @pytest.mark.parametrize(['label', 'image'], [(v, f'{v}.jpg') for v in _LABELS])
    def test_anime_dbaesthetic(self, label, image):
        image_file = get_testfile('dbaesthetic', image)
        assert anime_dbaesthetic(image_file, fmt='label') == label

    @pytest.mark.parametrize(['label', 'image'], [(v, f'{v}.jpg') for v in _LABELS])
    def test_anime_dbaesthetic_default(self, label, image):
        image_file = get_testfile('dbaesthetic', image)
        label_, percentile = anime_dbaesthetic(image_file)
        assert label_ == label

    @pytest.mark.parametrize(['label', 'image'], [(v, f'{v}.jpg') for v in _LABELS])
    def test_anime_dbaesthetic_list(self, label, image):
        image_file = get_testfile('dbaesthetic', image)
        r = anime_dbaesthetic(image_file, fmt=['label', 'percentile'])
        assert isinstance(r, list)
        assert len(r) == 2
        label_, percentile = r
        assert label_ == label

    @pytest.mark.parametrize(['label', 'image'], [(v, f'{v}.jpg') for v in _LABELS])
    def test_anime_dbaesthetic_dict(self, label, image):
        image_file = get_testfile('dbaesthetic', image)
        r = anime_dbaesthetic(image_file, fmt={'label': 'label', 'score': 'percentile', 'conf': 'confidence'})
        assert r['label'] == label
        assert isinstance(r['conf'], dict)
        assert len(r['conf']) == 7
