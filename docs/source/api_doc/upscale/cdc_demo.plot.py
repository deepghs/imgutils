import os

from huggingface_hub import HfFileSystem

from imgutils.upscale import upscale_with_cdc
from imgutils.upscale.cdc import _open_cdc_upscaler_model
from plot import image_plot

hf_fs = HfFileSystem()
repository = 'deepghs/cdc_anime_onnx'
_CDC_MODELS = [
    os.path.splitext(os.path.relpath(file, repository))[0]
    for file in hf_fs.glob(f'{repository}/*.onnx')
]

if __name__ == '__main__':
    demo_images = [
        ('sample/original.png', 'Small Logo'),
        ('sample/skadi.jpg', 'Illustration'),
        ('sample/hutao.png', 'Large Illustration'),
        # ('sample/xx.jpg', 'Illustration #2'),
        ('sample/rgba_restore.png', 'RGBA Artwork'),
    ]

    items = []
    for file, title in demo_images:
        items.append((file, title))
        for model in _CDC_MODELS:
            _, scale = _open_cdc_upscaler_model(model)
            items.append((upscale_with_cdc(file, model=model), f'{title}\n({scale}X By {model})'))

    image_plot(
        *items,
        columns=len(_CDC_MODELS) + 1,
        figsize=(4 * (len(_CDC_MODELS) + 1), 3.5 * len(demo_images)),
    )
