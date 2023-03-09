from ..data import load_image, ImageTyping

_GREYSCALE_K, _GREYSCALE_B = (-0.3578160565562811, 10.867076188331309)
_G_PNSR_THRESHOLD = -_GREYSCALE_B / _GREYSCALE_K


def is_greyscale(image: ImageTyping):
    from ..similarity import psnr

    image = load_image(image)
    return psnr(image, image.convert('L')) >= _G_PNSR_THRESHOLD
