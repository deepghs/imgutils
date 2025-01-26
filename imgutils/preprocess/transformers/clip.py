from PIL import Image

from imgutils.preprocess.pillow import PillowResize, PillowCenterCrop, PillowToTensor, PillowNormalize, PillowCompose, \
    PillowRescale, PillowConvertRGB

_DEFAULT_SIZE = {"shortest_edge": 224}
_DEFAULT_CROP_SIZE = {"height": 224, "width": 224}
_DEFAULT_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
_DEFAULT_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
_DEFAULT = object()


def create_clip_transforms(
        do_resize=True,
        size=_DEFAULT,
        resample=Image.BICUBIC,
        do_center_crop=True,
        crop_size=_DEFAULT,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=_DEFAULT,
        image_std=_DEFAULT,
        do_convert_rgb=True
):
    size = size if size is not _DEFAULT else _DEFAULT_SIZE
    crop_size = crop_size if crop_size is not _DEFAULT else _DEFAULT_CROP_SIZE
    image_mean = image_mean if image_mean is not _DEFAULT else _DEFAULT_IMAGE_MEAN
    image_std = image_std if image_std is not _DEFAULT else _DEFAULT_IMAGE_STD

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

    # Center crop
    if do_center_crop:
        transform_list.append(PillowCenterCrop((crop_size["height"], crop_size["width"])))

    # Convert to tensor (implicitly rescales to [0,1])
    transform_list.append(PillowToTensor())

    # Rescale (if different from 1/255)
    if do_rescale and rescale_factor != 1 / 255:
        transform_list.append(PillowRescale(rescale_factor * 255))

    # Normalize
    if do_normalize:
        transform_list.append(PillowNormalize(mean=image_mean, std=image_std))

    return PillowCompose(transform_list)


clip_transforms = create_clip_transforms(
    do_resize=True,
    size={"shortest_edge": 224},
    resample=Image.BICUBIC,
    do_center_crop=True,
    crop_size={"height": 224, "width": 224},
    do_rescale=True,
    rescale_factor=1 / 254,
    do_normalize=True,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
    do_convert_rgb=True
)
print(clip_transforms)
