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


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_DEFAULT = object()

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
