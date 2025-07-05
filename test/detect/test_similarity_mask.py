import numpy as np
import pytest

from imgutils.detect import calculate_mask_iou, masks_similarity, detection_with_mask_similarity
from imgutils.detect.similarity import _mask_to_bool_mask


# Fixtures for 2x2 masks
@pytest.fixture
def mask_2x2_1():
    return np.array([[1, 0], [0, 1]])


@pytest.fixture
def mask_2x2_2():
    return np.array([[0, 1], [1, 0]])


@pytest.fixture
def mask_2x2_3():
    return np.array([[1, 1], [0, 0]])


# Fixtures for 3x3 masks
@pytest.fixture
def mask_3x3_1():
    return np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])


@pytest.fixture
def mask_3x3_2():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


@pytest.fixture
def mask_3x3_3():
    return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


# Fixtures for 4x4 masks
@pytest.fixture
def mask_4x4_1():
    return np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])


@pytest.fixture
def mask_4x4_2():
    return np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])


@pytest.fixture
def mask_4x4_3():
    return np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])


@pytest.mark.unittest
class TestMaskFunctions:

    @pytest.mark.parametrize("mask, expected_output", [
        (np.array([[True, False], [False, True]]), np.array([[True, False], [False, True]])),
        (np.array([[False, True, False], [True, True, True], [False, True, False]]),
         np.array([[False, True, False], [True, True, True], [False, True, False]]),),
        (np.array([[True, True, True, True], [True, False, False, True],
                   [True, False, False, True], [True, True, True, True]]),
         np.array([[True, True, True, True], [True, False, False, True],
                   [True, False, False, True], [True, True, True, True]]),)
    ])
    def test_mask_to_bool_mask_raw(self, mask, expected_output):
        result = _mask_to_bool_mask(mask)
        np.testing.assert_array_equal(result, expected_output)

    @pytest.mark.parametrize("mask_fixture, expected_output", [
        ("mask_2x2_1", np.array([[True, False], [False, True]])),
        ("mask_3x3_2", np.array([[False, True, False], [True, True, True], [False, True, False]])),
        ("mask_4x4_3", np.array([[True, True, True, True], [True, False, False, True],
                                 [True, False, False, True], [True, True, True, True]]))
    ])
    def test_mask_to_bool_mask(self, request, mask_fixture, expected_output):
        mask = request.getfixturevalue(mask_fixture)
        result = _mask_to_bool_mask(mask)
        np.testing.assert_array_equal(result, expected_output)

    @pytest.mark.parametrize("mask1_fixture, mask2_fixture, expected_iou", [
        ("mask_2x2_1", "mask_2x2_2", 0.0),
        ("mask_3x3_1", "mask_3x3_2", 1 / 9),
        ("mask_4x4_1", "mask_4x4_2", 1 / 3)
    ])
    def test_calculate_mask_iou(self, request, mask1_fixture, mask2_fixture, expected_iou):
        mask1 = request.getfixturevalue(mask1_fixture)
        mask2 = request.getfixturevalue(mask2_fixture)
        iou = calculate_mask_iou(mask1, mask2)
        assert pytest.approx(iou, 0.01) == expected_iou

    @pytest.mark.parametrize("masks1_fixtures, masks2_fixtures, mode, expected_output", [
        (["mask_2x2_1", "mask_2x2_2"], ["mask_2x2_3"], "max", 1 / 3),
        (["mask_3x3_1", "mask_3x3_2"], ["mask_3x3_3"], "mean", 2 / 9),
        (["mask_4x4_1", "mask_4x4_2"], ["mask_4x4_3"], "raw", [3 / 7, 0.0])
    ])
    def test_masks_similarity(self, request, masks1_fixtures, masks2_fixtures, mode, expected_output):
        masks1 = [request.getfixturevalue(fixture) for fixture in masks1_fixtures]
        masks2 = [request.getfixturevalue(fixture) for fixture in masks2_fixtures]
        result = masks_similarity(masks1, masks2, mode)
        if isinstance(expected_output, list):
            assert pytest.approx(result, 0.01) == expected_output
        else:
            assert pytest.approx(result, 0.01) == expected_output

    @pytest.mark.parametrize("detect1_fixtures, detect2_fixtures, mode, expected_output", [
        ([("mask_2x2_1", "car"), ("mask_2x2_2", "person")],
         [("mask_2x2_3", "car"), ("mask_2x2_1", "person")], "mean", 1 / 6),
        ([("mask_3x3_2", "car"), ("mask_3x3_3", "person")],
         [("mask_3x3_1", "car"), ("mask_3x3_2", "person")], "max", 4 / 9),
        ([("mask_3x3_2", "car"), ("mask_3x3_3", "person")],
         [("mask_3x3_3", "person"), ("mask_3x3_2", "car")], "mean", 1.0),
        ([("mask_4x4_3", "car")], [("mask_4x4_1", "car"), ("mask_4x4_2", "person")],
         "raw", [3 / 7, 0.0])
    ])
    def test_detection_with_mask_similarity(self, request, detect1_fixtures, detect2_fixtures, mode, expected_output):
        detect1 = [(None, label, 0.9, request.getfixturevalue(fixture)) for fixture, label in detect1_fixtures]
        detect2 = [(None, label, 0.9, request.getfixturevalue(fixture)) for fixture, label in detect2_fixtures]
        result = detection_with_mask_similarity(detect1, detect2, mode)
        if isinstance(expected_output, list):
            assert pytest.approx(result, 0.01) == expected_output
        else:
            assert pytest.approx(result, 0.01) == expected_output

    def test_mask_to_bool_mask_errors(self):
        with pytest.raises(TypeError):
            _mask_to_bool_mask("not a numpy array")
        with pytest.raises(TypeError):
            _mask_to_bool_mask(np.array(["a", "b"]))

    def test_masks_similarity_edge_cases(self, request):
        assert masks_similarity([], [], mode='max') == 1.0
        assert masks_similarity([], [], mode='mean') == 1.0
        assert masks_similarity([], [], mode='raw') == []
        with pytest.raises(ValueError):
            masks_similarity(
                masks1=[
                    request.getfixturevalue('mask_4x4_1'),
                ],
                masks2=[
                    request.getfixturevalue('mask_4x4_2'),
                ],
                mode='invalid',
            )

    def test_detection_with_mask_similarity_edge_cases(self):
        assert detection_with_mask_similarity([], [], mode='mean') == 1.0
        with pytest.raises(ValueError):
            detection_with_mask_similarity([], [], mode='invalid')
