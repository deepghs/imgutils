from typing import Tuple

import numpy as np

BBoxTyping = Tuple[float, float, float, float]
BBoxWithScoreAndLabel = Tuple[BBoxTyping, str, float]
MaskWithScoreAndLabel = Tuple[BBoxTyping, str, float, np.ndarray]
