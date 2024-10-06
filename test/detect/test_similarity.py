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

    @pytest.mark.parametrize("bboxes1, bboxes2, mode, expected", [
        ([(0, 0, 2, 2), (3, 3, 5, 5)], [(1, 1, 3, 3), (4, 4, 6, 6)], 'mean', 0.14285714285714285),
        ([(0, 0, 2, 2), (3, 3, 5, 5)], [(1, 1, 3, 3), (4, 4, 6, 6)], 'max', 0.14285714285714285),
        ([(0, 0, 1, 1), (2, 2, 3, 3)], [(0.5, 0.5, 1.5, 1.5), (2.5, 2.5, 3.5, 3.5)], 'mean', 0.14285706122453645),
        ([(0, 0, 2, 2), (3, 3, 5, 5)], [(0.5, 0.5, 2.5, 2.5), (3.5, 3.5, 5.5, 5.5)], 'mean', 0.39130427977316873),
        ([(1, 1, 3, 3), (4, 4, 6, 6)], [(0, 0, 2, 2), (3, 3, 5, 5)], 'max', 0.14285714285714285),
        ([(0, 0, 1, 1), (2, 2, 4, 4)], [(0.25, 0.25, 1.25, 1.25), (2.25, 2.25, 4.25, 4.25)],
         'mean', 0.5057785572753247),
        ([(1, 1, 2, 2), (3, 3, 4, 4)], [(1.25, 1.25, 2.25, 2.25), (3.25, 3.25, 4.25, 4.25)],
         'mean', 0.3913040756145561),
        ([(0, 0, 3, 3), (4, 4, 7, 7)], [(1, 1, 4, 4), (5, 5, 8, 8)], 'mean', 0.2857142857142857),
        ([(1, 1, 4, 4), (5, 5, 8, 8)], [(0, 0, 3, 3), (4, 4, 7, 7)], 'max', 0.2857142857142857),
        ([(0, 0, 2, 2), (3, 3, 5, 5)], [(0.75, 0.75, 2.75, 2.75), (3.75, 3.75, 5.75, 5.75)],
         'mean', 0.24271840889811122),
    ])
    def test_bboxes_similarity(self, bboxes1, bboxes2, mode, expected):
        similarity = bboxes_similarity(bboxes1, bboxes2, mode)
        assert pytest.approx(similarity, 0.0001) == expected

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

    @pytest.mark.parametrize("detect1, detect2, mode, expected", [
        ([((0, 0, 2, 2), 'car', 0.9), ((3, 3, 5, 5), 'person', 0.8)],
         [((1, 1, 3, 3), 'car', 0.85), ((4, 4, 6, 6), 'person', 0.75)],
         'mean', 0.14285714285714285),
        ([((0, 0, 2, 2), 'car', 0.9), ((3, 3, 5, 5), 'person', 0.8)],
         [((1, 1, 3, 3), 'car', 0.85), ((4, 4, 6, 6), 'person', 0.75)],
         'max', 0.14285714285714285),
        ([((0, 0, 1, 1), 'bike', 0.7), ((2, 2, 3, 3), 'dog', 0.6)],
         [((0.5, 0.5, 1.5, 1.5), 'bike', 0.65), ((2.5, 2.5, 3.5, 3.5), 'dog', 0.55)],
         'mean', 0.14285706122453645),
        ([((0, 0, 2, 2), 'car', 0.9), ((3, 3, 5, 5), 'person', 0.8)],
         [((0.5, 0.5, 2.5, 2.5), 'car', 0.85), ((3.5, 3.5, 5.5, 5.5), 'person', 0.75)],
         'mean', 0.39130427977316873),
        ([((1, 1, 3, 3), 'car', 0.9), ((4, 4, 6, 6), 'person', 0.8)],
         [((0, 0, 2, 2), 'car', 0.85), ((3, 3, 5, 5), 'person', 0.75)],
         'max', 0.14285714285714285),
        ([((0, 0, 1, 1), 'bike', 0.7), ((2, 2, 4, 4), 'dog', 0.6)],
         [((0.25, 0.25, 1.25, 1.25), 'bike', 0.65), ((2.25, 2.25, 4.25, 4.25), 'dog', 0.55)],
         'mean', 0.5057785572753247),
        ([((1, 1, 2, 2), 'car', 0.9), ((3, 3, 4, 4), 'person', 0.8)],
         [((1.25, 1.25, 2.25, 2.25), 'car', 0.85), ((3.25, 3.25, 4.25, 4.25), 'person', 0.75)],
         'mean', 0.3913040756145561),
        ([((0, 0, 3, 3), 'car', 0.9), ((4, 4, 7, 7), 'person', 0.8)],
         [((1, 1, 4, 4), 'car', 0.85), ((5, 5, 8, 8), 'person', 0.75)],
         'mean', 0.2857142857142857),
        ([((1, 1, 4, 4), 'car', 0.9), ((5, 5, 8, 8), 'person', 0.8)],
         [((0, 0, 3, 3), 'car', 0.85), ((4, 4, 7, 7), 'person', 0.75)],
         'max', 0.2857142857142857),
        ([((0, 0, 2, 2), 'car', 0.9), ((3, 3, 5, 5), 'person', 0.8)],
         [((0.75, 0.75, 2.75, 2.75), 'car', 0.85), ((3.75, 3.75, 5.75, 5.75), 'person', 0.75)],
         'mean', 0.24271840889811122),
    ])
    def test_detection_similarity(self, detect1, detect2, mode, expected):
        similarity = detection_similarity(detect1, detect2, mode)
        assert pytest.approx(similarity, 0.0001) == expected
