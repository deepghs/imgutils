"""
This module provides functionality for creating image transformation pipelines compatible with CLIP (Contrastive Language-Image Pre-training) models.
It includes utilities for resizing, cropping, normalizing and converting images to the format expected by CLIP models.

The module integrates with the Hugging Face transformers library and provides compatibility with CLIP processors.
"""

from PIL import Image

from .base import _check_transformers, NotProcessorTypeError, register_creators_for_transformers, OPENAI_CLIP_MEAN, \
    OPENAI_CLIP_STD, _DEFAULT
from ..pillow import PillowResize, PillowCenterCrop, PillowToTensor, PillowNormalize, PillowCompose, PillowRescale, \
    PillowConvertRGB

_DEFAULT_SIZE = {"shortest_edge": 224}
_DEFAULT_CROP_SIZE = {"height": 224, "width": 224}


def create_clip_transforms(
        do_resize: bool = True,
        size=_DEFAULT,
        resample=Image.BICUBIC,
        do_center_crop=True,
        crop_size=_DEFAULT,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean=_DEFAULT,
        image_std=_DEFAULT,
        do_convert_rgb: bool = True
):
    """
    Creates a composition of image transforms typically used for CLIP models.

    :param do_resize: Whether to resize the image.
    :type do_resize: bool
    :param size: Target size for resizing. Can be {"shortest_edge": int} or {"height": int, "width": int}.
    :type size: dict
    :param resample: PIL resampling filter to use for resizing.
    :type resample: int
    :param do_center_crop: Whether to center crop the image.
    :type do_center_crop: bool
    :param crop_size: Size for center cropping in {"height": int, "width": int} format.
    :type crop_size: dict
    :param do_rescale: Whether to rescale pixel values.
    :type do_rescale: bool
    :param rescale_factor: Factor to use for rescaling pixels.
    :type rescale_factor: float
    :param do_normalize: Whether to normalize the image.
    :type do_normalize: bool
    :param image_mean: Mean values for normalization.
    :type image_mean: list or tuple
    :param image_std: Standard deviation values for normalization.
    :type image_std: list or tuple
    :param do_convert_rgb: Whether to convert image to RGB.
    :type do_convert_rgb: bool

    :return: A composed transformation pipeline.
    :rtype: PillowCompose
    """
    size = size if size is not _DEFAULT else _DEFAULT_SIZE
    crop_size = crop_size if crop_size is not _DEFAULT else _DEFAULT_CROP_SIZE
    image_mean = image_mean if image_mean is not _DEFAULT else OPENAI_CLIP_MEAN
    image_std = image_std if image_std is not _DEFAULT else OPENAI_CLIP_STD

    transform_list = []

    if do_convert_rgb:
        transform_list.append(PillowConvertRGB())

    if do_resize:
        if "shortest_edge" in size:
            transform_list.append(PillowResize(size["shortest_edge"], interpolation=resample))
        elif "height" in size and "width" in size:
            transform_list.append(PillowResize((size["height"], size["width"]), interpolation=resample))

    if do_center_crop:
        transform_list.append(PillowCenterCrop((crop_size["height"], crop_size["width"])))

    transform_list.append(PillowToTensor())

    if do_rescale and rescale_factor != 1 / 255:
        transform_list.append(PillowRescale(rescale_factor * 255))

    if do_normalize:
        transform_list.append(PillowNormalize(mean=image_mean, std=image_std))

    return PillowCompose(transform_list)


@register_creators_for_transformers()
def create_transforms_from_clip_processor(processor):
    """
    Creates image transforms from a CLIP processor configuration.

    :param processor: A CLIP processor or image processor instance from transformers library.
    :type processor: Union[CLIPProcessor, CLIPImageProcessor]

    :return: A composed transformation pipeline matching the processor's configuration.
    :rtype: PillowCompose
    :raises NotProcessorTypeError: If the provided processor is not a CLIP processor.
    """
    _check_transformers()
    from transformers import CLIPProcessor, CLIPImageProcessor

    if isinstance(processor, CLIPProcessor):
        processor = processor.image_processor
    elif isinstance(processor, CLIPImageProcessor):
        pass
    else:
        raise NotProcessorTypeError(f'Unknown CLIP processor - {processor!r}.')
    processor: CLIPImageProcessor

    return create_clip_transforms(
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
