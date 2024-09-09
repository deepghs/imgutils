from typing import Optional

import piexif
from piexif.helper import UserComment

from ..data import ImageTyping, load_image


def read_geninfo_parameters(image: ImageTyping) -> Optional[str]:
    image = load_image(image, mode=None, force_background=None)
    infos = image.info or {}
    return infos.get('parameters')


def read_geninfo_exif(image: ImageTyping) -> Optional[str]:
    image = load_image(image, mode=None, force_background=None)
    infos = image.info or {}
    if "exif" in infos:
        exif_data = infos["exif"]
        try:
            exif = piexif.load(exif_data)
        except OSError:
            # memory / exif was not valid so piexif tried to read from a file
            exif = None

        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        try:
            exif_comment = UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode("utf8", errors="ignore")

        return exif_comment
    else:
        return None


def read_geninfo_gif(image: ImageTyping) -> Optional[str]:
    image = load_image(image, mode=None, force_background=None)
    infos = image.info or {}
    if "comment" in infos:  # for gif
        return infos["comment"].decode("utf8", errors="ignore")
    else:
        return None
