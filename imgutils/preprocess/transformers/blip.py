from PIL import Image

from .base import OPENAI_CLIP_STD, OPENAI_CLIP_MEAN, _DEFAULT, _check_transformers, NotProcessorTypeError, \
    register_creators_for_transformers
from ..pillow import PillowConvertRGB, PillowRescale, PillowNormalize, PillowToTensor, PillowResize, PillowCompose

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
    size = size if size is not _DEFAULT else _DEFAULT_SIZE
    image_mean = image_mean if image_mean is not _DEFAULT else OPENAI_CLIP_MEAN
    image_std = image_std if image_std is not _DEFAULT else OPENAI_CLIP_STD

    transform_list = []

    # Convert to RGB if needed
    if do_convert_rgb:
        transform_list.append(PillowConvertRGB())

    # Resize if needed
    if do_resize:
        transform_list.append(PillowResize((size["height"], size["width"]), interpolation=resample))

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
