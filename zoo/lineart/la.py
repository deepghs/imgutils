import numpy as np
import torch
from controlnet_aux import LineartDetector
from controlnet_aux.open_pose import HWC3, resize_image
from einops import rearrange

from .onnx import _BaseOnnxMixin


class _MyLineartDetector(LineartDetector, _BaseOnnxMixin):
    def _get_model(self, coarse=False, **kwargs):
        return self.model_coarse if coarse else self.model

    def preprocess(self, input_image, coarse=False, detect_resolution: int = 512, **kwargs):
        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        image = input_image
        with torch.no_grad():
            image = torch.from_numpy(image).float().to(device)
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')

            return image
