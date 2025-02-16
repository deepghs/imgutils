import pytest
from PIL import Image

from imgutils.preprocess.transformers import is_valid_size_dict, convert_to_size_dict, get_size_dict


@pytest.fixture
def sample_image():
    return Image.new('RGB', (100, 200))


@pytest.fixture
def valid_size_dicts():
    return [
        {"height": 100, "width": 100},
        {"shortest_edge": 100},
        {"shortest_edge": 100, "longest_edge": 200},
        {"longest_edge": 200},
        {"max_height": 100, "max_width": 200},
    ]


@pytest.fixture
def invalid_size_dicts():
    return [
        {},
        {"height": 100},
        {"width": 100},
        {"unknown": 100},
        {"height": 100, "width": 100, "extra": 100},
    ]


@pytest.mark.unittest
class TestSizeDictFunctions:
    def test_is_valid_size_dict_with_valid_inputs(self, valid_size_dicts):
        for size_dict in valid_size_dicts:
            assert is_valid_size_dict(size_dict)

    def test_is_valid_size_dict_with_invalid_inputs(self, invalid_size_dicts):
        for size_dict in invalid_size_dicts:
            assert not is_valid_size_dict(size_dict)

    def test_is_valid_size_dict_with_non_dict_input(self):
        assert not is_valid_size_dict(None)
        assert not is_valid_size_dict([])
        assert not is_valid_size_dict(100)
        assert not is_valid_size_dict("string")

    def test_convert_to_size_dict_with_int_square(self):
        result = convert_to_size_dict(100, default_to_square=True)
        assert result == {"height": 100, "width": 100}

    def test_convert_to_size_dict_with_int_non_square(self):
        result = convert_to_size_dict(100, default_to_square=False)
        assert result == {"shortest_edge": 100}

    def test_convert_to_size_dict_with_int_and_max_size(self):
        result = convert_to_size_dict(100, max_size=200, default_to_square=False)
        assert result == {"shortest_edge": 100, "longest_edge": 200}

    def test_convert_to_size_dict_with_tuple_height_width_order(self):
        result = convert_to_size_dict((100, 200), height_width_order=True)
        assert result == {"height": 100, "width": 200}

    def test_convert_to_size_dict_with_tuple_width_height_order(self):
        result = convert_to_size_dict((100, 200), height_width_order=False)
        assert result == {"height": 200, "width": 100}

    def test_convert_to_size_dict_with_list(self):
        result = convert_to_size_dict([100, 200], height_width_order=True)
        assert result == {"height": 100, "width": 200}

    def test_convert_to_size_dict_with_none_and_max_size(self):
        result = convert_to_size_dict(None, max_size=200, default_to_square=False)
        assert result == {"longest_edge": 200}

    def test_convert_to_size_dict_invalid_combinations(self):
        with pytest.raises(ValueError):
            convert_to_size_dict(100, max_size=200, default_to_square=True)

        with pytest.raises(ValueError):
            convert_to_size_dict(None, max_size=200, default_to_square=True)

        with pytest.raises(ValueError):
            convert_to_size_dict("invalid")

    def test_get_size_dict_with_direct_dict(self):
        size_dict = {"height": 100, "width": 200}
        result = get_size_dict(size_dict)
        assert result == size_dict

    def test_get_size_dict_with_int(self):
        result = get_size_dict(100)
        assert result == {"height": 100, "width": 100}

    def test_get_size_dict_with_tuple(self):
        result = get_size_dict((100, 200))
        assert result == {"height": 100, "width": 200}

    def test_get_size_dict_with_invalid_dict(self):
        with pytest.raises(ValueError):
            get_size_dict({"invalid": 100})

    def test_get_size_dict_with_none_and_max_size(self):
        result = get_size_dict(None, max_size=200, default_to_square=False)
        assert result == {"longest_edge": 200}

    def test_get_size_dict_custom_param_name(self):
        with pytest.raises(ValueError) as exc_info:
            get_size_dict({"invalid": 100}, param_name="custom_param")
        assert "custom_param must have one of the following" in str(exc_info.value)
