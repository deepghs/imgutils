"""
This module provides utilities for manipulating nested data structures, particularly focusing on
value replacement and name extraction from complex nested structures like lists, tuples, and dictionaries.

The module offers functionality to recursively traverse nested data structures and either replace values
based on a mapping or extract unique names/values from the structure.
"""

from typing import List

__all__ = [
    'vreplace',
    'vnames',
]


def vreplace(v, mapping):
    """
    Recursively replaces values in a nested data structure using a mapping dictionary.

    This function traverses through nested lists, tuples, and dictionaries, replacing values
    according to the provided mapping. If a value exists as a key in the mapping dictionary,
    it will be replaced with its corresponding value.

    :param v: The input data structure to process (can be a nested structure of lists, tuples, dicts)
    :type v: Any
    :param mapping: A dictionary defining value replacements
    :type mapping: dict

    :return: A new data structure with values replaced according to the mapping
    :rtype: Any

    :example:
        >>> data = {'a': [1, 2, 3], 'b': {'x': 1}}
        >>> mapping = {1: 'one', 2: 'two'}
        >>> vreplace(data, mapping)
        {'a': ['one', 'two', 3], 'b': {'x': 'one'}}
    """
    if isinstance(v, (list, tuple)):
        return type(v)([vreplace(vitem, mapping) for vitem in v])
    elif isinstance(v, dict):
        return type(v)({key: vreplace(value, mapping) for key, value in v.items()})
    else:
        try:
            _ = hash(v)
        except TypeError:  # pragma: no cover
            return v
        else:
            return mapping.get(v, v)


def _v_iternames(v):
    """
    Internal helper function that yields all hashable values from a nested data structure.

    :param v: The input data structure to traverse
    :type v: Any

    :yield: Hashable values found in the data structure
    :rtype: Generator
    """
    if isinstance(v, (list, tuple)):
        for item in v:
            yield from _v_iternames(item)
    elif isinstance(v, dict):
        for _, item in v.items():
            yield from _v_iternames(item)
    else:
        try:
            _ = hash(v)
        except TypeError:  # pragma: no cover
            pass
        else:
            yield v


def vnames(v, str_only: bool = True) -> List[str]:
    """
    Extracts unique values/names from a nested data structure.

    This function traverses through the input data structure and collects all unique
    hashable values. When str_only is True, it only collects string values.

    :param v: The input data structure to process
    :type v: Any
    :param str_only: If True, only string values are collected
    :type str_only: bool

    :return: A list of unique values found in the data structure
    :rtype: List[str]

    :example:
        >>> data = {'a': ['x', 'y', 1], 'b': {'z': 'x'}}
        >>> vnames(data)
        ['x', 'y', 'z']
        >>> vnames(data, str_only=False)
        ['x', 'y', 1, 'z']
    """
    name_set = set()
    for name in _v_iternames(v):
        if not str_only or isinstance(name, str):
            name_set.add(name)
    return list(name_set)
