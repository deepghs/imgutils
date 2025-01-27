from PIL import Image

from .base import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, _DEFAULT, _check_transformers, NotProcessorTypeError, \
    register_creators_for_transformers
from ..pillow import PillowCompose, PillowNormalize, PillowRescale, PillowToTensor, PillowResize, PillowConvertRGB

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
        transforms_list.append(PillowResize((size["height"], size["width"]), interpolation=resample))

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
