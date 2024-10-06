import pytest

from imgutils.detect.similarity import calculate_iou, bboxes_similarity, detection_similarity


@pytest.fixture
def sample_bboxes():
    return [
        (0, 0, 10, 10),
        (5, 5, 15, 15),
        (20, 20, 30, 30),
    ]


@pytest.fixture
def sample_detections():
    return [
        ((0, 0, 10, 10), 'car', 0.9),
        ((5, 5, 15, 15), 'person', 0.8),
        ((20, 20, 30, 30), 'car', 0.7),
    ]


@pytest.mark.unittest
class TestBBoxFunctions:
    def test_calculate_iou(self):
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        assert calculate_iou(box1, box2) == pytest.approx(25.0 / 175)

    def test_bboxes_similarity_max(self, sample_bboxes):
        result = bboxes_similarity(sample_bboxes, sample_bboxes, mode='max')
        assert isinstance(result, float)
        assert result == pytest.approx(1.0)

    def test_bboxes_similarity_mean(self, sample_bboxes):
        result = bboxes_similarity(sample_bboxes, sample_bboxes, mode='mean')
        assert isinstance(result, float)
        assert result == pytest.approx(1.0)

    def test_bboxes_similarity_raw(self, sample_bboxes):
        result = bboxes_similarity(sample_bboxes, sample_bboxes, mode='raw')
        assert isinstance(result, list)
        assert result == pytest.approx([1.0, 1.0, 1.0])

    def test_bboxes_similarity_invalid_mode(self, sample_bboxes):
        with pytest.raises(ValueError, match="Unknown similarity mode for bboxes - 'invalid'"):
            bboxes_similarity(sample_bboxes, sample_bboxes, mode='invalid')

    def test_bboxes_similarity_unequal_length(self, sample_bboxes):
        with pytest.raises(ValueError, match="Length of bboxes lists not match"):
            bboxes_similarity(sample_bboxes, sample_bboxes[:-1])

    def test_detection_similarity_max(self, sample_detections):
        result = detection_similarity(sample_detections, sample_detections, mode='max')
        assert isinstance(result, float)
        assert result == pytest.approx(1.0)

    def test_detection_similarity_mean(self, sample_detections):
        result = detection_similarity(sample_detections, sample_detections, mode='mean')
        assert isinstance(result, float)
        assert result == pytest.approx(1.0)

    def test_detection_similarity_raw(self, sample_detections):
        result = detection_similarity(sample_detections, sample_detections, mode='raw')
        assert isinstance(result, list)
        assert result == pytest.approx([1.0, 1.0, 1.0])

    def test_detection_similarity_invalid_mode(self, sample_detections):
        with pytest.raises(ValueError, match="Unknown similarity mode for bboxes - 'invalid'"):
            detection_similarity(sample_detections, sample_detections, mode='invalid')

    def test_detection_similarity_unequal_length(self, sample_detections):
        with pytest.raises(ValueError, match="Length of bboxes not match on label"):
            detection_similarity(sample_detections, sample_detections[:-1])
