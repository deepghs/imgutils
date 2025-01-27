"""
Transformers Integration Module

This module provides functionality for integrating with the transformers library,
particularly for image processing tasks. It includes constants for standard image
normalization values and utilities for creating image transforms from transformers
processors.
"""

try:
    import transformers
except (ImportError, ModuleNotFoundError):
    _HAS_TRANSFORMERS = False
else:
    _HAS_TRANSFORMERS = True


def _check_transformers():
    """
    Check if transformers library is available.

    :raises EnvironmentError: If transformers is not installed
    """
    if not _HAS_TRANSFORMERS:
        raise EnvironmentError('No torchvision available.\n'
                               'Please install it by `pip install dghs-imgutils[transformers]`.')


# Standard normalization constants
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_DEFAULT = object()


class NotProcessorTypeError(TypeError):
    """
    Exception raised when a processor type is not recognized or supported.

    This error occurs when attempting to create transforms from an unsupported
    or unknown transformers processor type.
    """
    pass


_FN_CREATORS = []


def register_creators_for_transformers():
    """
    Decorator for registering transform creator functions.

    This decorator adds the decorated function to the list of available
    transform creators that will be tried when creating transforms from
    a transformers processor.

    :return: Decorator function
    :rtype: callable

    :example:

        >>> @register_creators_for_transformers()
        >>> def my_transform_creator(processor):
        ...     # Create and return transforms
        ...     pass
    """

    def _decorator(func):
        _FN_CREATORS.append(func)
        return func

    return _decorator


def create_transforms_from_transformers(processor):
    """
    Create image transforms from a transformers processor.

    This function attempts to create appropriate image transforms by trying
    each registered creator function until one succeeds.

    :param processor: A transformers processor object
    :type processor: object
    :return: Image transforms appropriate for the given processor
    :rtype: object
    :raises NotProcessorTypeError: If no suitable creator is found for the processor

    :example:
        >>> from transformers import AutoImageProcessor
        >>> from imgutils.preprocess.transformers import create_transforms_from_transformers
        >>>
        >>> processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> transforms = create_transforms_from_transformers(processor)
        >>> transforms
        PillowCompose(
            PillowConvertRGB(force_background='white')
            PillowResize(size=224, interpolation=bicubic, max_size=None, antialias=True)
            PillowCenterCrop(size=(224, 224))
            PillowToTensor()
            PillowNormalize(mean=[0.48145467 0.4578275  0.40821072], std=[0.26862955 0.2613026  0.2757771 ])
        )
    """
    for _fn in _FN_CREATORS:
        try:
            return _fn(processor)
        except NotProcessorTypeError:
            pass
    else:
        raise NotProcessorTypeError(f'Unknown transformers processor - {processor!r}.')
