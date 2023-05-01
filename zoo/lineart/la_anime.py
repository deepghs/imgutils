import cv2
import numpy as np
import torch
from controlnet_aux import LineartAnimeDetector
from controlnet_aux.open_pose import HWC3, resize_image
from einops import rearrange

from imgutils.data import load_image
from .onnx import _FixedOnnxMixin


class _MyLineartAnimeDetector(LineartAnimeDetector, _FixedOnnxMixin):
    def __init__(self, model):
        LineartAnimeDetector.__init__(self, model)
        _FixedOnnxMixin.__init__(self, model)

    def preprocess(self, input_image, detect_resolution=512):
        input_image = load_image(input_image)
        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        height, width, channels = input_image.shape
        height_n = 256 * int(np.ceil(float(height) / 256.0))
        weight_n = 256 * int(np.ceil(float(width) / 256.0))
        img = cv2.resize(input_image, (weight_n, height_n), interpolation=cv2.INTER_CUBIC)
        with torch.no_grad():
            image_feed = torch.from_numpy(img).float().to(device)
            image_feed = image_feed / 127.5 - 1.0
            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')

            return image_feed

    def postprocess(self, input_image, output):
        line = output[0, 0] * 127.5 + 127.5
        line = line.cpu().numpy()

        height, width, channels = input_image.shape
        line = cv2.resize(line, (width, height), interpolation=cv2.INTER_CUBIC)
        line = line.clip(0, 255).astype(np.uint8)

        detected_map = HWC3(line)
        detected_map = cv2.resize(detected_map, (width, height), interpolation=cv2.INTER_LINEAR)
        detected_map = 255 - detected_map

        return detected_map
