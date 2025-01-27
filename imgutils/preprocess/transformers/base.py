try:
    import transformers
except (ImportError, ModuleNotFoundError):
    _HAS_TRANSFORMERS = False
else:
    _HAS_TRANSFORMERS = True


def _check_transformers():
    if not _HAS_TRANSFORMERS:
        raise EnvironmentError('No torchvision available.\n'
                               'Please install it by `pip install dghs-imgutils[transformers]`.')


class NotProcessorTypeError(TypeError):
    pass


_FN_CREATORS = []


def register_creators_for_transformers():
    def _decorator(func):
        _FN_CREATORS.append(func)
        return func

    return _decorator


def create_transforms_from_transformers(processor):
    for _fn in _FN_CREATORS:
        try:
            return _fn(processor)
        except NotProcessorTypeError:
            pass
    else:
        raise NotProcessorTypeError(f'Unknown transformers processor - {processor!r}.')
