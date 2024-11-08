import numpy as np
import pytest

from imgutils.utils import sigmoid


@pytest.fixture
def sample_input():
    return np.array([-1, 0, 1])


@pytest.fixture
def expected_output():
    return np.array([0.26894142, 0.5, 0.73105858])


@pytest.mark.unittest
class TestUtilsFuncSigmoid:
    def test_sigmoid_scalar(self):
        assert np.isclose(sigmoid(0), 0.5)

    def test_sigmoid_array(self, sample_input, expected_output):
        result = sigmoid(sample_input)
        np.testing.assert_array_almost_equal(result, expected_output)

    def test_sigmoid_large_positive(self):
        assert np.isclose(sigmoid(100), 1.0)

    def test_sigmoid_large_negative(self):
        assert np.isclose(sigmoid(-100), 0.0)

    def test_sigmoid_zero(self):
        assert sigmoid(0) == 0.5

    def test_sigmoid_type(self):
        assert isinstance(sigmoid(1), float)
        assert isinstance(sigmoid(np.array([1])), np.ndarray)

    def test_sigmoid_shape(self):
        input_array = np.array([[1, 2], [3, 4]])
        result = sigmoid(input_array)
        assert result.shape == input_array.shape
