from PIL import Image

from .base import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, _DEFAULT, register_creators_for_transformers, \
    _check_transformers, NotProcessorTypeError
from ..pillow import PillowRescale, PillowResize, PillowToTensor, PillowNormalize, PillowCompose

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
    # Initialize default values
    size = size if size is not _DEFAULT else _DEFAULT_SIZE
    image_mean = image_mean if image_mean is not _DEFAULT else IMAGENET_DEFAULT_MEAN
    image_std = image_std if image_std is not _DEFAULT else IMAGENET_DEFAULT_STD

    transform_list = []

    # Add resize transform if enabled
    if do_resize:
        transform_list.append(
            PillowResize(
                (size["height"], size["width"]),
                interpolation=resample
            )
        )

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
