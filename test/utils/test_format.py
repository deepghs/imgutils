import pytest

from imgutils.utils import vnames, vreplace


@pytest.fixture
def sample_list():
    return [1, 2, 3, "a", "b"]


@pytest.fixture
def sample_dict():
    return {"x": 1, "y": "a", "z": [2, "b", {"w": 3}]}


@pytest.fixture
def sample_mapping():
    return {1: "one", "a": "A", 3: "three"}


@pytest.mark.unittest
class TestVReplaceFunctions:
    def test_vreplace_list(self, sample_list, sample_mapping):
        result = vreplace(sample_list, sample_mapping)
        assert result == ["one", 2, "three", "A", "b"]
        assert isinstance(result, list)

    def test_vreplace_tuple(self, sample_mapping):
        input_tuple = (1, "a", 3)
        result = vreplace(input_tuple, sample_mapping)
        assert result == ("one", "A", "three")
        assert isinstance(result, tuple)

    def test_vreplace_dict(self, sample_dict, sample_mapping):
        result = vreplace(sample_dict, sample_mapping)
        assert result == {"x": "one", "y": "A", "z": [2, "b", {"w": "three"}]}
        assert isinstance(result, dict)

    def test_vreplace_scalar(self, sample_mapping):
        assert vreplace(1, sample_mapping) == "one"
        assert vreplace(4, sample_mapping) == 4  # unmapped value

    def test_vreplace_empty_mapping(self, sample_list):
        result = vreplace(sample_list, {})
        assert result == sample_list

    def test_vnames_list(self, sample_list):
        result = vnames(sample_list)
        assert set(result) == {"a", "b"}

        result_all = vnames(sample_list, str_only=False)
        assert set(result_all) == {1, 2, 3, "a", "b"}

    def test_vnames_dict(self, sample_dict):
        result = vnames(sample_dict)
        assert set(result) == {"a", "b"}

        result_all = vnames(sample_dict, str_only=False)
        assert set(result_all) == {1, 2, 3, "a", "b"}

    def test_vnames_empty(self):
        assert set(vnames([])) == set()
        assert set(vnames({})) == set()
        assert set(vnames([], str_only=False)) == set()

    def test_vnames_nested(self):
        nested = [1, ["a", 2, ["b", 3]], {"x": "c"}]
        assert set(vnames(nested)) == {"a", "b", "c"}
        assert set(vnames(nested, str_only=False)) == {1, 2, 3, "a", "b", "c"}

    def test_vnames_mixed_types(self):
        mixed = [1.5, True, None, "test"]
        assert set(vnames(mixed)) == {"test"}
        assert set(vnames(mixed, str_only=False)) == {1.5, True, None, "test"}
