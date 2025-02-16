from PIL import Image

from ..pillow import PillowResize

VALID_SIZE_DICT_KEYS = (
    {"height", "width"},
    {"shortest_edge"},
    {"shortest_edge", "longest_edge"},
    {"longest_edge"},
    {"max_height", "max_width"},
)


def is_valid_size_dict(size_dict):
    return isinstance(size_dict, dict) and any(set(size_dict.keys()) == keys for keys in VALID_SIZE_DICT_KEYS)


def convert_to_size_dict(size, max_size=None, default_to_square=True, height_width_order=True):
    if isinstance(size, int):
        if default_to_square:
            if max_size is not None:
                raise ValueError("Cannot specify both size as an int, with default_to_square=True and max_size")
            return {"height": size, "width": size}
        size_dict = {"shortest_edge": size}
        if max_size is not None:
            size_dict["longest_edge"] = max_size
        return size_dict

    if isinstance(size, (tuple, list)):
        if height_width_order:
            return {"height": size[0], "width": size[1]}
        return {"height": size[1], "width": size[0]}

    if size is None and max_size is not None:
        if default_to_square:
            raise ValueError("Cannot specify both default_to_square=True and max_size")
        return {"longest_edge": max_size}

    raise ValueError(f"Could not convert size input to size dict: {size}")


def get_size_dict(
        size=None,
        max_size=None,
        height_width_order=True,
        default_to_square=True,
        param_name="size",
) -> dict:
    """
    Converts size parameter into a standardized dictionary format.

    Args:
        size: Input size as int, tuple/list, or dict
        max_size: Optional maximum size constraint
        height_width_order: If True, tuple order is (height,width)
        default_to_square: If True, single int creates square output
        param_name: Parameter name for error messages

    Returns:
        Dictionary with standardized size format
    """
    if not isinstance(size, dict):
        size_dict = convert_to_size_dict(size, max_size, default_to_square, height_width_order)
    else:
        size_dict = size

    if not is_valid_size_dict(size_dict):
        raise ValueError(
            f"{param_name} must have one of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size_dict.keys()}"
        )
    return size_dict


def _create_resize(size, resample=Image.BICUBIC):
    """
    Create a PillowResize transform based on the given size configuration.

    :param size: Dictionary containing size configuration, either with 'shortest_edge'
                or both 'height' and 'width' keys
    :type size: dict
    :param resample: PIL resampling filter to use for resizing, defaults to Image.BICUBIC
    :type resample: int

    :return: A PillowResize transform configured according to the size parameters
    :rtype: PillowResize

    :raises ValueError: If the size configuration is not recognized
    """
    size = get_size_dict(size)
    if "shortest_edge" in size:
        extra_keys = sorted(set(size.keys()) - {'shortest_edge'})
        if extra_keys:
            raise ValueError(f'Unsupported resize dict - {size!r}.')
        return PillowResize(size["shortest_edge"], interpolation=resample)
    elif "height" in size and "width" in size:
        return PillowResize((size["height"], size["width"]), interpolation=resample)
    else:
        raise ValueError(f'Unknown size configuration - {size!r}.')  # pragma: no cover
