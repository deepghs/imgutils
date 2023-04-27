import numpy as np
import torch
from controlnet_aux import PidiNetDetector
from controlnet_aux.open_pose import HWC3, resize_image
from einops import rearrange

from .onnx import _FixedOnnxMixin


class _MyPidiNetDetector(PidiNetDetector, _FixedOnnxMixin):
    def __init__(self, model):
        PidiNetDetector.__init__(self, model)
        _FixedOnnxMixin.__init__(self, model)

    def preprocess(self, input_image, detect_resolution: int = 512):
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(input_image).float()
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')

            return image_pidi
