"""
This module provides image transformation utilities specifically designed for SigLIP (Simple Grouped Learning with Image-text Pairs) models.
It includes functions to create image transformation pipelines compatible with the SigLIP architecture, supporting operations like
resizing, rescaling, normalization, and RGB conversion.
"""

from PIL import Image

from .base import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, _DEFAULT, _check_transformers, NotProcessorTypeError, \
    register_creators_for_transformers
from .size import _create_resize
from ..pillow import PillowCompose, PillowNormalize, PillowRescale, PillowToTensor, PillowConvertRGB

_DEFAULT_SIZE = {"height": 224, "width": 224}


def create_siglip_transforms(
        do_resize: bool = True,
        size=_DEFAULT,
        resample: int = Image.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean=_DEFAULT,
        image_std=_DEFAULT,
        do_convert_rgb: bool = True,
):
    """
    Creates a composition of image transformations for SigLIP model input processing.

    This function builds a pipeline of image transformations that can include:

    - RGB conversion
    - Image resizing
    - Tensor conversion
    - Image rescaling
    - Normalization

    :param do_resize: Whether to resize the image
    :type do_resize: bool
    :param size: Target size dictionary with 'height' and 'width' keys
    :type size: dict
    :param resample: PIL image resampling filter to use for resizing
    :type resample: int
    :param do_rescale: Whether to rescale pixel values
    :type do_rescale: bool
    :param rescale_factor: Factor to use for pixel value rescaling
    :type rescale_factor: float
    :param do_normalize: Whether to normalize the image
    :type do_normalize: bool
    :param image_mean: Mean values for normalization
    :type image_mean: tuple or list
    :param image_std: Standard deviation values for normalization
    :type image_std: tuple or list
    :param do_convert_rgb: Whether to convert image to RGB
    :type do_convert_rgb: bool

    :return: A composed transformation pipeline
    :rtype: PillowCompose
    """
    # Set default values
    size = size if size is not _DEFAULT else _DEFAULT_SIZE
    image_mean = image_mean if image_mean is not _DEFAULT else IMAGENET_STANDARD_MEAN
    image_std = image_std if image_std is not _DEFAULT else IMAGENET_STANDARD_STD

    transforms_list = []

    # Convert to RGB
    if do_convert_rgb:
        transforms_list.append(PillowConvertRGB())

    # Resize
    if do_resize:
        transforms_list.append(_create_resize(size, resample=resample))

    # Convert to tensor (implicitly rescales to 0-1)
    transforms_list.append(PillowToTensor())

    # Rescale if needed (only if different from 1/255)
    if do_rescale and rescale_factor != 1 / 255:
        transforms_list.append(PillowRescale(rescale_factor * 255))

    # Normalize
    if do_normalize:
        transforms_list.append(PillowNormalize(mean=image_mean, std=image_std))

    return PillowCompose(transforms_list)


@register_creators_for_transformers()
def create_transforms_from_siglip_processor(processor):
    """
    Creates image transformations from a SigLIP processor configuration.

    This function extracts transformation parameters from a HuggingFace SigLIP
    image processor and creates a corresponding transformation pipeline.

    :param processor: A HuggingFace SigLIP image processor instance
    :type processor: SiglipImageProcessor

    :return: A composed transformation pipeline
    :rtype: PillowCompose
    :raises NotProcessorTypeError: If the processor is not a SiglipImageProcessor
    """
    _check_transformers()
    from transformers import SiglipImageProcessor

    if isinstance(processor, SiglipImageProcessor):
        pass
    else:
        raise NotProcessorTypeError(f'Unknown Siglip processor - {processor!r}.')
    processor: SiglipImageProcessor

    return create_siglip_transforms(
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
