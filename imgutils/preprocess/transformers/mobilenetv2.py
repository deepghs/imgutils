"""
MobileNetV2 transforms module for creating image transformations compatible with MobileNetV2 models.

This module provides functions to create compositions of image transforms that replicate
the behavior of the MobileNetV2ImageProcessor from the transformers library. The main
components include:

The module is designed to work with PIL images and provide compatibility with
transformer-based vision models while using native Python image processing.
"""

from typing import Dict, List, Optional, Union

from PIL import Image

from .base import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, _DEFAULT, register_creators_for_transformers, \
    _check_transformers, NotProcessorTypeError
from .size import get_size_dict, _create_resize
from ..pillow import PillowCenterCrop, PillowNormalize, PillowCompose, PillowToTensor, PillowRescale

_DEFAULT_SIZE = {"shortest_edge": 256}
_DEFAULT_CROP_SIZE = {"height": 224, "width": 224}


def create_mobilenetv2_transforms(
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = _DEFAULT,
        resample=Image.BILINEAR,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = _DEFAULT,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = _DEFAULT,
        image_std: Optional[Union[float, List[float]]] = _DEFAULT,
):
    """
    Creates a composition of transforms that replicates the behavior of MobileNetV2ImageProcessor.

    This function builds a pipeline of image transformations typically used for MobileNetV2 models,
    including resizing, center cropping, tensor conversion, rescaling, and normalization.

    :param do_resize: Whether to resize the image.
    :type do_resize: bool

    :param size: Size dictionary specifying resize parameters. Can include keys like
                 'shortest_edge', 'height', 'width', etc.
    :type size: Optional[Dict[str, int]]

    :param resample: Resampling filter to use for resizing operations.
    :type resample: PIL.Image.Resampling

    :param do_center_crop: Whether to apply center cropping to the image.
    :type do_center_crop: bool

    :param crop_size: Dictionary specifying the height and width for center cropping.
    :type crop_size: Dict[str, int]

    :param do_rescale: Whether to rescale pixel values after tensor conversion.
    :type do_rescale: bool

    :param rescale_factor: Factor by which to rescale the image pixel values.
    :type rescale_factor: Union[int, float]

    :param do_normalize: Whether to normalize the image with mean and std.
    :type do_normalize: bool

    :param image_mean: Mean values for normalization, per channel.
    :type image_mean: Optional[Union[float, List[float]]]

    :param image_std: Standard deviation values for normalization, per channel.
    :type image_std: Optional[Union[float, List[float]]]

    :return: A composition of transforms matching MobileNetV2ImageProcessor behavior.
    :rtype: PillowCompose
    """
    transform_list = []

    # Set defaults if not provided
    size = size if size is not None else _DEFAULT_SIZE
    size = get_size_dict(size, default_to_square=False)

    crop_size = crop_size if crop_size is not None else _DEFAULT_CROP_SIZE
    crop_size = get_size_dict(crop_size, param_name="crop_size")

    image_mean = image_mean if image_mean is not _DEFAULT else IMAGENET_DEFAULT_MEAN
    image_std = image_std if image_std is not _DEFAULT else IMAGENET_DEFAULT_STD

    # Add resize transform if requested
    if do_resize:
        transform_list.append(_create_resize(size, resample=resample))

    # Add center crop transform if requested
    if do_center_crop:
        transform_list.append(PillowCenterCrop((crop_size["height"], crop_size["width"])))

    # Add to_tensor transform to convert PIL image to tensor (0-255 to 0-1 range)
    transform_list.append(PillowToTensor())

    # Add rescale transform if requested and if not already handled by ToTensor
    if do_rescale and rescale_factor != 1 / 255:
        transform_list.append(PillowRescale(rescale_factor * 255))

    # Add normalize transform if requested
    if do_normalize:
        transform_list.append(PillowNormalize(mean=image_mean, std=image_std))

    return PillowCompose(transform_list)


@register_creators_for_transformers()
def create_transforms_from_mobilenetv2_processor(processor):
    """
    Creates transform composition from a MobileNetV2ImageProcessor instance.

    This function extracts configuration from a transformers MobileNetV2ImageProcessor
    and creates an equivalent transform pipeline using the create_mobilenetv2_transforms
    function.

    :param processor: A MobileNetV2ImageProcessor instance from the transformers library.
    :type processor: transformers.MobileNetV2ImageProcessor

    :return: A composition of transforms matching the processor's configuration.
    :rtype: PillowCompose

    :raises NotProcessorTypeError: If the provided processor is not a MobileNetV2ImageProcessor.
    """
    _check_transformers()
    from transformers import MobileNetV2ImageProcessor

    if isinstance(processor, MobileNetV2ImageProcessor):
        pass
    else:
        raise NotProcessorTypeError(f'Unknown mobilenetv2 processor - {processor!r}.')
    processor: MobileNetV2ImageProcessor

    return create_mobilenetv2_transforms(
        do_resize=processor.do_resize,
        size=processor.size,
        do_center_crop=processor.do_center_crop,
        crop_size=processor.crop_size,
        resample=processor.resample,
        do_rescale=processor.do_rescale,
        rescale_factor=processor.rescale_factor,
        do_normalize=processor.do_normalize,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
    )
