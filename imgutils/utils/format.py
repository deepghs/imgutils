__all__ = [
    'vreplace',
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
