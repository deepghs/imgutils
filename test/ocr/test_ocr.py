import pytest
from PIL import Image

from imgutils.ocr import detect_text_with_ocr, list_det_models, list_rec_models, ocr
from test.testings import get_testfile


@pytest.fixture()
def ocr_img_plot():
    yield get_testfile('ocr', 'plot.png')


@pytest.fixture()
def ocr_img_plot_pil(ocr_img_plot):
    yield Image.open(ocr_img_plot)


@pytest.fixture()
def ocr_img_comic():
    yield get_testfile('ocr', 'comic.jpg')


@pytest.fixture()
def ocr_img_comic_pil(ocr_img_comic):
    yield Image.open(ocr_img_comic)


@pytest.fixture()
def ocr_img_anime_subtitle():
    yield get_testfile('ocr', 'anime_subtitle.jpg')


@pytest.fixture()
def ocr_img_anime_subtitle_pil(ocr_img_anime_subtitle):
    yield Image.open(ocr_img_anime_subtitle)


@pytest.fixture()
def ocr_img_post_text():
    yield get_testfile('ocr', 'post_text.jpg')


@pytest.fixture()
def ocr_img_post_text_pil(ocr_img_post_text):
    yield Image.open(ocr_img_post_text)


@pytest.mark.unittest
class TestOcr:
    def test_detect_text_with_ocr_comic(self, ocr_img_comic):
        detections = detect_text_with_ocr(ocr_img_comic)
        assert len(detections) == 8

        values = []
        for bbox, label, score in detections:
            assert label == 'text'
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((742, 485, 809, 511), 0.954),
            ((682, 98, 734, 124), 0.93),
            ((716, 136, 836, 164), 0.904),
            ((144, 455, 196, 485), 0.874),
            ((719, 455, 835, 488), 0.862),
            ((124, 478, 214, 508), 0.848),
            ((1030, 557, 1184, 578), 0.835),
            ((427, 129, 553, 154), 0.824)
        ])

    def test_detect_text_with_ocr_anime_subtitle(self, ocr_img_anime_subtitle_pil):
        detections = detect_text_with_ocr(ocr_img_anime_subtitle_pil)
        assert len(detections) == 2

        values = []
        for bbox, label, score in detections:
            assert label == 'text'
            values.append((bbox, int(score * 1000) / 1000))

        assert values == pytest.approx([
            ((312, 567, 690, 600), 0.817),
            ((332, 600, 671, 636), 0.798)
        ])

    def test_list_det_models(self):
        lst = list_det_models()
        assert 'ch_PP-OCRv4_det' in lst
        assert 'ch_ppocr_mobile_v2.0_det' in lst
        assert 'en_PP-OCRv3_det' in lst

    def test_ocr_comic(self, ocr_img_comic):
        detections = ocr(ocr_img_comic)
        assert len(detections) == 8

        assert detections == pytest.approx([
            ((742, 485, 809, 511), 'MOB.', 0.9356705927336156),
            ((716, 136, 836, 164), 'SHISHOU,', 0.8933000384412466),
            ((682, 98, 734, 124), 'BUT', 0.8730931912907247),
            ((144, 455, 196, 485), 'OH,', 0.8417627579351514),
            ((427, 129, 553, 154), 'A MIRROR.', 0.7366019454049503),
            ((1030, 557, 1184, 578), '(EL)  GATO IBERICO', 0.7271127306351021),
            ((719, 455, 835, 488), "THAt'S △", 0.701928390168364),
            ((124, 478, 214, 508), 'LOOK!', 0.6965972578194936),
        ], abs=1e-3)

    def test_ocr_plot(self, ocr_img_plot):
        detections = ocr(ocr_img_plot)
        assert len(detections) >= 75

    def test_list_rec_models(self):
        lst = list_rec_models()
        assert 'arabic_PP-OCRv3_rec' in lst
        assert 'ch_PP-OCRv4_rec' in lst
        assert 'ch_ppocr_mobile_v2.0_rec' in lst
        assert 'japan_PP-OCRv3_rec' in lst
        assert 'latin_PP-OCRv3_rec' in lst
        assert 'korean_PP-OCRv3_rec' in lst
        assert 'cyrillic_PP-OCRv3_rec' in lst
