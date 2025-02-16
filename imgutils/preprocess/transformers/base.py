"""
Transformers Integration Module for Image Processing

This module provides functionality for integrating with the Hugging Face transformers library,
particularly focused on image processing tasks. It includes standard image normalization
constants and utilities for creating image transforms from transformers processors.

Usage:
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

try:
    import transformers
except (ImportError, ModuleNotFoundError):
    _HAS_TRANSFORMERS = False
else:
    _HAS_TRANSFORMERS = True


def _check_transformers():
    """
    Check if the transformers library is available in the current environment.

    :raises EnvironmentError: If transformers is not installed, with instructions for installation
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
    Exception raised when an unsupported processor type is encountered.

    This custom exception is used when the system cannot create transforms
    from a given transformers processor, either because the processor type
    is not recognized or is not supported by any registered transform creators.

    :inherits: TypeError
    """
    pass


_FN_CREATORS = []


def register_creators_for_transformers():
    """
    Decorator that registers functions as transform creators for transformers processors.

    This decorator system allows for extensible support of different processor types.
    When a function is decorated with this decorator, it is added to the list of
    available transform creators that will be tried when creating transforms from
    a transformers processor.

    :return: Decorator function that registers the decorated function
    :rtype: callable

    :example:
        >>> @register_creators_for_transformers()
        >>> def create_clip_transforms(processor):
        ...     if not hasattr(processor, 'feature_extractor'):
        ...         raise NotProcessorTypeError()
        ...     # Create and return transforms for CLIP
        ...     return transforms
    """

    def _decorator(func):
        _FN_CREATORS.append(func)
        return func

    return _decorator


def create_transforms_from_transformers(processor):
    """
    Create appropriate image transforms from a given transformers processor.

    This function attempts to create image transforms by iterating through
    registered creator functions until one successfully creates transforms
    for the given processor type.

    :param processor: A processor instance from the transformers library
    :type processor: transformers.ImageProcessor or similar

    :return: A composition of image transforms suitable for the given processor
    :rtype: PillowCompose or similar transform object

    :raises NotProcessorTypeError: If no registered creator can handle the processor type

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
