"""
This module provides functionality for creating image transformation pipelines specifically for Vision Transformer (ViT) models.
It includes functions to create transforms for image preprocessing tasks like resizing, rescaling, normalization and tensor conversion.
The transforms are compatible with both custom usage and Hugging Face's transformers library ViT processors.

The module supports creating transform pipelines that match the preprocessing steps used in ViT models,
ensuring images are properly prepared for model inference.
"""

from PIL import Image

from .base import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, _DEFAULT, register_creators_for_transformers, \
    _check_transformers, NotProcessorTypeError
from .size import _create_resize
from ..pillow import PillowRescale, PillowToTensor, PillowNormalize, PillowCompose

_DEFAULT_SIZE = {"height": 224, "width": 224}


def create_vit_transforms(
        do_resize: bool = True,
        size=_DEFAULT,
        resample: int = Image.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean=_DEFAULT,
        image_std=_DEFAULT,
):
    """
    Create a composition of image transforms typically used for ViT models.

    This function creates a transform pipeline that can include resizing, tensor conversion,
    rescaling, and normalization operations. The transforms are applied in sequence to
    prepare images for ViT model input.

    :param do_resize: Whether to resize the input images
    :type do_resize: bool
    :param size: Target size for resizing, should be dict with 'height' and 'width' keys
    :type size: dict
    :param resample: PIL resampling filter to use for resizing
    :type resample: int
    :param do_rescale: Whether to rescale pixel values
    :type do_rescale: bool
    :param rescale_factor: Factor to use for rescaling pixel values
    :type rescale_factor: float
    :param do_normalize: Whether to normalize the image
    :type do_normalize: bool
    :param image_mean: Mean values for normalization
    :type image_mean: tuple or list
    :param image_std: Standard deviation values for normalization
    :type image_std: tuple or list

    :return: A composition of image transforms
    :rtype: PillowCompose
    """
    # Initialize default values
    size = size if size is not _DEFAULT else _DEFAULT_SIZE
    image_mean = image_mean if image_mean is not _DEFAULT else IMAGENET_DEFAULT_MEAN
    image_std = image_std if image_std is not _DEFAULT else IMAGENET_DEFAULT_STD

    transform_list = []

    # Add resize transform if enabled
    if do_resize:
        transform_list.append(_create_resize(size, resample=resample))

    # Convert to tensor (always needed)
    transform_list.append(PillowToTensor())

    # Add rescaling if enabled
    # Note: ToTensor already scales to [0,1], so we only need additional scaling if factor != 1/255
    if do_rescale and rescale_factor != 1 / 255:
        transform_list.append(PillowRescale(rescale_factor * 255))

    # Add normalization if enabled0
    if do_normalize:
        transform_list.append(PillowNormalize(mean=image_mean, std=image_std))

    return PillowCompose(transform_list)


@register_creators_for_transformers()
def create_transforms_from_vit_processor(processor):
    """
    Create image transforms from a Hugging Face ViT processor configuration.

    This function takes a ViT image processor from the transformers library and creates
    a matching transform pipeline that replicates the processor's preprocessing steps.

    :param processor: A ViT image processor from Hugging Face transformers
    :type processor: ViTImageProcessor

    :return: A composition of image transforms matching the processor's configuration
    :rtype: PillowCompose
    :raises NotProcessorTypeError: If the provided processor is not a ViTImageProcessor
    """
    _check_transformers()
    from transformers import ViTImageProcessor

    if isinstance(processor, ViTImageProcessor):
        pass
    else:
        raise NotProcessorTypeError(f'Unknown ViT processor - {processor!r}.')
    processor: ViTImageProcessor

    return create_vit_transforms(
        do_resize=processor.do_resize,
        size=processor.size,
        resample=processor.resample,
        do_rescale=processor.do_rescale,
        rescale_factor=processor.rescale_factor,
        do_normalize=processor.do_normalize,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
    )
