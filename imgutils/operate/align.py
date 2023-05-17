from imgutils.data import ImageTyping, load_image


def align(image: ImageTyping, max_size: int):
    image = load_image(image, force_background=None)
    width, height = image.size

    r = max_size / max(width, height)
    new_width, new_heihgt = int(width * r), int(height * r)
    return image.resize((new_width, new_heihgt))
