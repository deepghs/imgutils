"""
This module provides functionality for creating image transformation pipelines specifically designed for ConvNext models.
It includes utilities for resizing, rescaling, normalizing and converting images to tensors, with special handling for
ConvNext's image processing requirements.

The module integrates with the Hugging Face Transformers library and provides compatibility with ConvNextImageProcessor.
"""

from PIL import Image

from .base import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, _DEFAULT, register_creators_for_transformers, \
    _check_transformers, NotProcessorTypeError
from ..pillow import PillowRescale, PillowResize, PillowCenterCrop, PillowToTensor, PillowNormalize, PillowCompose

_DEFAULT_SIZE = {"shortest_edge": 384}
_DEFAULT_CROP_PCT = 224 / 256


def create_convnext_transforms(
        do_resize: bool = True,
        size=_DEFAULT,
        crop_pct: float = _DEFAULT,
        resample=Image.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean=_DEFAULT,
        image_std=_DEFAULT,
):
    """
    Create a composition of image transforms specifically tailored for ConvNext models.

    This function creates a transformation pipeline that can include resizing, rescaling,
    and normalization operations. The transforms are applied in the following order:

    1. Resize (optional)
    2. Convert to tensor
    3. Rescale (optional)
    4. Normalize (optional)

    :param do_resize: Whether to resize the image
    :type do_resize: bool
    :param size: Target size dictionary with 'shortest_edge' key
    :type size: dict
    :param crop_pct: Center crop percentage, used to compute resize size
    :type crop_pct: float
    :param resample: PIL resampling filter to use for resizing
    :type resample: int
    :param do_rescale: Whether to rescale pixel values
    :type do_rescale: bool
    :param rescale_factor: Factor to use for rescaling pixels
    :type rescale_factor: float
    :param do_normalize: Whether to normalize the image
    :type do_normalize: bool
    :param image_mean: Mean values for normalization
    :type image_mean: tuple or list
    :param image_std: Standard deviation values for normalization
    :type image_std: tuple or list

    :return: A composed transformation pipeline
    :rtype: PillowCompose
    """
    size = size if size is not _DEFAULT else _DEFAULT_SIZE
    crop_pct = crop_pct if crop_pct is not _DEFAULT else _DEFAULT_CROP_PCT
    image_mean = image_mean if image_mean is not _DEFAULT else IMAGENET_STANDARD_MEAN
    image_std = image_std if image_std is not _DEFAULT else IMAGENET_STANDARD_STD

    transform_list = []

    if do_resize:
        shortest_edge = size["shortest_edge"]
        if shortest_edge < 384:
            resize_shortest_edge = int(shortest_edge / crop_pct)
            transform_list.extend([
                PillowResize(resize_shortest_edge, interpolation=resample),
                PillowCenterCrop(shortest_edge)
            ])
        else:
            transform_list.append(PillowResize((shortest_edge, shortest_edge), interpolation=resample))

    transform_list.append(PillowToTensor())

    if do_rescale and rescale_factor != 1 / 255:
        transform_list.append(PillowRescale(rescale_factor * 255))

    if do_normalize:
        transform_list.append(PillowNormalize(mean=image_mean, std=image_std))

    return PillowCompose(transform_list)


@register_creators_for_transformers()
def create_transforms_from_convnext_processor(processor):
    """
    Create image transforms from a ConvNext processor configuration.

    This function takes a Hugging Face ConvNextImageProcessor and creates a corresponding
    transformation pipeline that matches its configuration settings.

    :param processor: The ConvNext image processor to create transforms from
    :type processor: ConvNextImageProcessor

    :return: A composed transformation pipeline matching the processor's configuration
    :rtype: PillowCompose
    :raises NotProcessorTypeError: If the provided processor is not a ConvNextImageProcessor
    """
    _check_transformers()
    from transformers import ConvNextImageProcessor

    if isinstance(processor, ConvNextImageProcessor):
        pass
    else:
        raise NotProcessorTypeError(f'Unknown ConvNext processor - {processor!r}.')
    processor: ConvNextImageProcessor

    return create_convnext_transforms(
        do_resize=processor.do_resize,
        size=processor.size,
        crop_pct=processor.crop_pct,
        resample=processor.resample,
        do_rescale=processor.do_rescale,
        rescale_factor=processor.rescale_factor,
        do_normalize=processor.do_normalize,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
    )
