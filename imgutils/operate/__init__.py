from .align import align_maxsize
from .censor_ import register_censor_method, censor_areas, censor_nsfw, BaseCensor, ColorCensor, BlurCensor, \
    PixelateCensor
from .imgcensor import ImageBasedCensor, EmojiBasedCensor
from .squeeze import squeeze, squeeze_with_transparency
