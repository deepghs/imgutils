import numpy as np
import torch
from controlnet_aux import HEDdetector
from controlnet_aux.open_pose import resize_image, HWC3
from einops import rearrange

from .onnx import _FixedOnnxMixin


class _MyHEDDetector(HEDdetector, _FixedOnnxMixin):
    def __init__(self, model):
        HEDdetector.__init__(self, model)
        _FixedOnnxMixin.__init__(self, model)

    def preprocess(self, input_image, detect_resolution: int = 512):
        device = next(iter(self.netNetwork.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image).float()
            image_hed = image_hed.to(device)
            image_hed = image_hed / 255.0
            image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
            return image_hed
