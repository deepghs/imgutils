"""
This module provides image transformation utilities specifically designed for BLIP (Bootstrapping Language-Image Pre-training) models.
It includes functions to create transformation pipelines for processing images according to BLIP's requirements.

The transformations include operations like resizing, RGB conversion, normalization, and tensor conversion,
all implemented using Pillow-based operations.
"""

from PIL import Image

from .base import OPENAI_CLIP_STD, OPENAI_CLIP_MEAN, _DEFAULT, _check_transformers, NotProcessorTypeError, \
    register_creators_for_transformers
from .size import _create_resize
from ..pillow import PillowConvertRGB, PillowRescale, PillowNormalize, PillowToTensor, PillowCompose

_DEFAULT_SIZE = {"height": 384, "width": 384}


def create_blip_transforms(
        do_resize: bool = True,
        size=_DEFAULT,
        resample=Image.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean=_DEFAULT,
        image_std=_DEFAULT,
        do_convert_rgb: bool = True,
):
    """
    Create a transformation pipeline for BLIP image processing.

    This function builds a sequence of image transformations commonly used in BLIP models,
    including RGB conversion, resizing, tensor conversion, rescaling, and normalization.

    :param do_resize: Whether to resize the image.
    :type do_resize: bool
    :param size: Target size for resizing, expects dict with 'height' and 'width' keys.
                Defaults to {'height': 384, 'width': 384}.
    :type size: dict
    :param resample: Resampling filter for resize operation. Defaults to PIL.Image.BICUBIC.
    :type resample: int
    :param do_rescale: Whether to rescale pixel values.
    :type do_rescale: bool
    :param rescale_factor: Factor to rescale pixel values. Defaults to 1/255.
    :type rescale_factor: float
    :param do_normalize: Whether to normalize the image.
    :type do_normalize: bool
    :param image_mean: Mean values for normalization. Defaults to OPENAI_CLIP_MEAN.
    :type image_mean: tuple or list
    :param image_std: Standard deviation values for normalization. Defaults to OPENAI_CLIP_STD.
    :type image_std: tuple or list
    :param do_convert_rgb: Whether to convert image to RGB.
    :type do_convert_rgb: bool

    :return: A composed transformation pipeline.
    :rtype: PillowCompose
    """
    size = size if size is not _DEFAULT else _DEFAULT_SIZE
    image_mean = image_mean if image_mean is not _DEFAULT else OPENAI_CLIP_MEAN
    image_std = image_std if image_std is not _DEFAULT else OPENAI_CLIP_STD

    transform_list = []

    # Convert to RGB if needed
    if do_convert_rgb:
        transform_list.append(PillowConvertRGB())

    # Resize if needed
    if do_resize:
        transform_list.append(_create_resize(size, resample=resample))

    # Convert PIL to tensor (which automatically scales to [0,1])
    transform_list.append(PillowToTensor())

    # If you do_rescale is True, but we don't want the automatic [0,1] scaling of ToTensor
    if do_rescale and rescale_factor != 1 / 255:
        transform_list.append(PillowRescale(rescale_factor * 255))

    # Normalize if needed
    if do_normalize:
        transform_list.append(PillowNormalize(mean=image_mean, std=image_std))

    return PillowCompose(transform_list)


@register_creators_for_transformers()
def create_transforms_from_blip_processor(processor):
    """
    Create image transformations from a HuggingFace BLIP processor.

    This function extracts configuration from a HuggingFace BLIP processor and creates
    a corresponding transformation pipeline using create_blip_transforms.

    :param processor: A HuggingFace BLIP image processor instance.
    :type processor: transformers.BlipImageProcessor

    :return: A composed transformation pipeline configured according to the processor's settings.
    :rtype: PillowCompose
    :raises NotProcessorTypeError: If the provided processor is not a BlipImageProcessor.
    """
    _check_transformers()
    from transformers import BlipImageProcessor

    if isinstance(processor, BlipImageProcessor):
        pass
    else:
        raise NotProcessorTypeError(f'Unknown blip processor - {processor!r}.')
    processor: BlipImageProcessor

    return create_blip_transforms(
        do_resize=processor.do_resize,
        size=processor.size,
        resample=processor.resample,
        do_rescale=processor.do_rescale,
        rescale_factor=processor.rescale_factor,
        do_normalize=processor.do_normalize,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
        do_convert_rgb=processor.do_convert_rgb,
    )
