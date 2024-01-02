import pytest
from PIL import Image

from imgutils.ocr import detect_text_with_ocr, list_det_models, list_rec_models, ocr
from imgutils.ocr.detect import _open_ocr_detection_model
from imgutils.ocr.recognize import _open_ocr_recognition_dictionary, _open_ocr_recognition_model
from test.testings import get_testfile


@pytest.fixture(autouse=True, scope='module')
def _clear_cache():
    try:
        yield
    finally:
        _open_ocr_detection_model.cache_clear()
        _open_ocr_recognition_model.cache_clear()
        _open_ocr_recognition_dictionary.cache_clear()


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


@pytest.fixture()
def ocr_img_cn_text():
    yield get_testfile('ocr', 'cn_text.png')


@pytest.fixture()
def ocr_img_cn_text_pil(ocr_img_cn_text):
    yield Image.open(ocr_img_cn_text)


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

        bboxes = []
        texts = []
        scores = []
        for bbox, text, score in detections:
            bboxes.append(bbox)
            texts.append(text)
            scores.append(score)

        assert bboxes == pytest.approx([
            (742, 485, 809, 511),
            (716, 136, 836, 164),
            (682, 98, 734, 124),
            (144, 455, 196, 485),
            (427, 129, 553, 154),
            (1030, 557, 1184, 578),
            (719, 455, 835, 488),
            (124, 478, 214, 508),
        ])
        assert texts == ['MOB.', 'SHISHOU,', 'BUT', 'OH,', 'A MIRROR.', '(EL)  GATO IBERICO', "THAt'S △", 'LOOK!']
        assert scores == pytest.approx([
            0.9356677655964869,
            0.8932994278321376,
            0.8730925493136663,
            0.8417598172118067,
            0.7365999885917329,
            0.7271122893745091,
            0.7019268051682541,
            0.6965953319577997
        ], abs=1e-3)

    def test_ocr_plot(self, ocr_img_plot):
        detections = ocr(ocr_img_plot)
        assert len(detections) >= 75

    def test_ocr_cn_text(self, ocr_img_cn_text):
        detections = ocr(ocr_img_cn_text)
        assert len(detections) >= 25

        bboxes = []
        texts = []
        scores = []
        for bbox, text, score in detections:
            bboxes.append(bbox)
            texts.append(text)
            scores.append(score)

        assert '算法列表' in texts
        assert '算法名' in texts
        assert '训练数据集' in texts
        assert '年份' in texts
        assert '任务' in texts
        assert 'word_acc' in texts
        assert 'SVTR' in texts

    def test_list_rec_models(self):
        lst = list_rec_models()
        assert 'arabic_PP-OCRv3_rec' in lst
        assert 'ch_PP-OCRv4_rec' in lst
        assert 'ch_ppocr_mobile_v2.0_rec' in lst
        assert 'japan_PP-OCRv3_rec' in lst
        assert 'latin_PP-OCRv3_rec' in lst
        assert 'korean_PP-OCRv3_rec' in lst
        assert 'cyrillic_PP-OCRv3_rec' in lst
