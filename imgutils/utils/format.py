from typing import List

__all__ = [
    'vreplace',
    'vnames',
]


def vreplace(v, mapping):
    """
    Replaces values in a data structure using a mapping dictionary.
    :param v: The input data structure.
    :type v: Any
    :param mapping: A dictionary mapping values to replacement values.
    :type mapping: Dict
    :return: The modified data structure.
    :rtype: Any
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
    name_set = set()
    for name in _v_iternames(v):
        if not str_only or isinstance(name, str):
            name_set.add(name)
    return list(name_set)
