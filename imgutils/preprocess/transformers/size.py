"""
Image resizing configuration and processing module.

This module provides utilities for handling and standardizing image size specifications
in various formats. It supports multiple ways to specify image sizes including fixed dimensions,
aspect ratio preservation, and maximum size constraints.

Size dictionary formats supported:
    - {"height": h, "width": w} : Exact dimensions
    - {"shortest_edge": s} : Preserve aspect ratio with given shortest edge
    - {"shortest_edge": s, "longest_edge": l} : Constrain both edges
    - {"longest_edge": l} : Maximum size while preserving aspect ratio
    - {"max_height": h, "max_width": w} : Independent height/width constraints
"""

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
    """
    Validate if a dictionary contains valid image size specifications.

    :param size_dict: Dictionary to validate
    :type size_dict: dict

    :return: True if the dictionary contains valid size specifications, False otherwise
    :rtype: bool

    :examples:
        >>> is_valid_size_dict({"height": 100, "width": 200})
        True
        >>> is_valid_size_dict({"shortest_edge": 100})
        True
        >>> is_valid_size_dict({"invalid_key": 100})
        False
    """
    return isinstance(size_dict, dict) and any(set(size_dict.keys()) == keys for keys in VALID_SIZE_DICT_KEYS)


def convert_to_size_dict(size, max_size=None, default_to_square=True, height_width_order=True):
    """
    Convert various size input formats to a standardized size dictionary.

    :param size: Size specification as integer, tuple/list, or None
    :type size: int or tuple or list or None
    :param max_size: Optional maximum size constraint
    :type max_size: int or None
    :param default_to_square: If True, single integer creates square dimensions
    :type default_to_square: bool
    :param height_width_order: If True, tuple values are (height, width), else (width, height)
    :type height_width_order: bool

    :return: Dictionary with standardized size format
    :rtype: dict

    :raises ValueError: If size specification is invalid or incompatible with other parameters

    :examples:
        >>> convert_to_size_dict(100)
        {'height': 100, 'width': 100}
        >>> convert_to_size_dict((200, 300), height_width_order=True)
        {'height': 200, 'width': 300}
        >>> convert_to_size_dict(100, max_size=200, default_to_square=False)
        {'shortest_edge': 100, 'longest_edge': 200}
    """
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
    Convert and validate size parameters into a standardized dictionary format.

    This function serves as the main entry point for size processing, handling various
    input formats and ensuring they conform to valid size specifications.

    :param size: Size specification as integer, tuple/list, dictionary, or None
    :type size: int or tuple or list or dict or None
    :param max_size: Optional maximum size constraint
    :type max_size: int or None
    :param height_width_order: If True, tuple values are (height, width), else (width, height)
    :type height_width_order: bool
    :param default_to_square: If True, single integer creates square dimensions
    :type default_to_square: bool
    :param param_name: Parameter name for error messages
    :type param_name: str

    :return: Dictionary with standardized size format
    :rtype: dict

    :raises ValueError: If size specification is invalid or incompatible with other parameters

    :examples:
        >>> get_size_dict(100)
        {'height': 100, 'width': 100}
        >>> get_size_dict({'shortest_edge': 100})
        {'shortest_edge': 100}
        >>> get_size_dict((200, 300), height_width_order=True)
        {'height': 200, 'width': 300}
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

    This internal function creates a resize transformation that respects the specified
    size constraints while maintaining aspect ratio when appropriate.

    :param size: Dictionary containing size configuration
    :type size: dict
    :param resample: PIL resampling filter to use for resizing
    :type resample: int

    :return: Configured resize transformation object
    :rtype: PillowResize

    :raises ValueError: If the size configuration is invalid or not recognized

    :examples:
        >>> transform = _create_resize({'shortest_edge': 100})
        >>> transform = _create_resize({'height': 200, 'width': 300})
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
