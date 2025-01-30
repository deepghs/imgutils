"""
This module provides image transformation utilities for BiT (Big Transfer) models.
It includes functions for creating image preprocessing pipelines that can handle
operations like resizing, cropping, normalization and RGB conversion.
The module is designed to work with both standalone usage and Hugging Face's transformers
library integration.
"""
from PIL import Image

from .base import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, _DEFAULT, register_creators_for_transformers, _check_transformers, \
    NotProcessorTypeError
from ..pillow import PillowConvertRGB, PillowResize, PillowCenterCrop, PillowToTensor, PillowNormalize, PillowCompose, \
    PillowRescale

_DEFAULT_SIZE = {"shortest_edge": 224}
_DEFAULT_CROP_SIZE = {"height": 224, "width": 224}


def create_bit_transforms(
        do_resize: bool = True,
        size=_DEFAULT,
        resample=Image.BICUBIC,
        do_center_crop: bool = True,
        crop_size=_DEFAULT,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean=_DEFAULT,
        image_std=_DEFAULT,
        do_convert_rgb: bool = True,
):
    """
    Create an image transformation pipeline for BiT models.

    This function creates a composition of image transformations including RGB conversion,
    resizing, center cropping, tensor conversion, rescaling and normalization.

    :param do_resize: Whether to resize the image.
    :type do_resize: bool
    :param size: Target size for resizing. Can be {"shortest_edge": int} or {"height": int, "width": int}.
    :type size: dict
    :param resample: PIL interpolation method for resizing.
    :type resample: int
    :param do_center_crop: Whether to perform center cropping.
    :type do_center_crop: bool
    :param crop_size: Size for center cropping, in format {"height": int, "width": int}.
    :type crop_size: dict
    :param do_rescale: Whether to rescale pixel values.
    :type do_rescale: bool
    :param rescale_factor: Factor to rescale pixel values.
    :type rescale_factor: float
    :param do_normalize: Whether to normalize the image.
    :type do_normalize: bool
    :param image_mean: Mean values for normalization.
    :type image_mean: list or tuple
    :param image_std: Standard deviation values for normalization.
    :type image_std: list or tuple
    :param do_convert_rgb: Whether to convert image to RGB.
    :type do_convert_rgb: bool

    :return: A composition of image transformations.
    :rtype: PillowCompose
    :raises ValueError: If size configuration is invalid.
    """
    # Set default values
    size = size if size is not _DEFAULT else _DEFAULT_SIZE
    crop_size = crop_size if crop_size is not _DEFAULT else _DEFAULT_CROP_SIZE
    image_mean = image_mean if image_mean is not _DEFAULT else OPENAI_CLIP_MEAN
    image_std = image_std if image_std is not _DEFAULT else OPENAI_CLIP_STD

    transform_list = []

    # Convert to RGB
    if do_convert_rgb:
        transform_list.append(PillowConvertRGB())

    # Resize
    if do_resize:
        if "shortest_edge" in size:
            transform_list.append(PillowResize(size["shortest_edge"], interpolation=resample))
        elif "height" in size and "width" in size:
            transform_list.append(PillowResize((size["height"], size["width"]), interpolation=resample))
        else:
            raise ValueError(f'Unknown size configuration - {size!r}.')  # pragma: no cover

    # Center crop
    if do_center_crop:
        transform_list.append(PillowCenterCrop((crop_size["height"], crop_size["width"])))

    # Convert to tensor (implicitly scales to [0,1])
    transform_list.append(PillowToTensor())

    # Rescale
    if do_rescale and rescale_factor != 1 / 255:
        transform_list.append(PillowRescale(rescale_factor * 255))

    # Normalize
    if do_normalize:
        transform_list.append(PillowNormalize(mean=image_mean, std=image_std))

    return PillowCompose(transform_list)


@register_creators_for_transformers()
def create_transforms_from_bit_processor(processor):
    """
    Create image transformations from a BiT image processor.

    This function creates a transformation pipeline based on the configuration
    of a Hugging Face BitImageProcessor.

    :param processor: The BiT image processor to create transforms from.
    :type processor: BitImageProcessor
    :return: A composition of image transformations.
    :rtype: PillowCompose
    :raises NotProcessorTypeError: If the processor is not a BitImageProcessor.
    """
    _check_transformers()
    from transformers import BitImageProcessor

    if isinstance(processor, BitImageProcessor):
        pass
    else:
        raise NotProcessorTypeError(f'Unknown Bit processor - {processor!r}.')
    processor: BitImageProcessor

    return create_bit_transforms(
        do_resize=processor.do_resize,
        size=processor.size,
        resample=processor.resample,
        do_center_crop=processor.do_center_crop,
        crop_size=processor.crop_size,
        do_rescale=processor.do_rescale,
        rescale_factor=processor.rescale_factor,
        do_normalize=processor.do_normalize,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
        do_convert_rgb=processor.do_convert_rgb,
    )
