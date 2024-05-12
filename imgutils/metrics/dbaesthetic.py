"""
Overview:
    A tool for assessing the aesthetic quality of anime images using a pre-trained model,
    based on danbooru dataset and metadata analysis result of
    `HakuBooru <https://github.com/KohakuBlueleaf/HakuBooru>`_ by
    `KohakuBlueleaf <https://github.com/KohakuBlueleaf>`_.

    .. image:: dbaesthetic_full.plot.py.svg
        :align: center

    This is an overall benchmark of all the operations in aesthetic models:

    .. image:: dbaesthetic_benchmark.plot.py.svg
        :align: center

"""
from typing import Dict, Optional, Tuple

import numpy as np
from huggingface_hub import hf_hub_download

from ..data import ImageTyping
from ..generic import ClassifyModel
from ..utils import vreplace

__all__ = [
    'anime_dbaesthetic',
]

_DEFAULT_MODEL_NAME = 'swinv2pv3_v0_448_ls0.2_x'
_REPO_ID = 'deepghs/anime_aesthetic'
_LABELS = ["worst", "low", "normal", "good", "great", "best", "masterpiece"]
_DEFAULT_LABEL_MAPPING = {
    'masterpiece': 0.95,
    'best': 0.85,
    'great': 0.75,
    'good': 0.5,
    'normal': 0.25,
    'low': 0.1,
    'worst': 0.0,
}


class AestheticModel:
    """
    A model for assessing the aesthetic quality of anime images.
    """

    def __init__(self, repo_id: str):
        """
        Initializes an AestheticModel instance.

        :param repo_id: The repository ID of the aesthetic assessment model.
        :type repo_id: str
        """
        self.repo_id = repo_id
        self.classifier = ClassifyModel(repo_id)
        self.cached_samples: Dict[str, Tuple] = {}

    def get_aesthetic_score(self, image: ImageTyping, model_name: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the aesthetic score and confidence for an anime image.

        :param image: The input anime image.
        :type image: ImageTyping
        :param model_name: The name of the aesthetic assessment model to use.
        :type model_name: str
        :return: A tuple containing the aesthetic score and confidence.
        :rtype: Tuple[float, Dict[str, float]]
        """
        scores = self.classifier.predict_score(image, model_name)
        return sum(scores[label] * i for i, label in enumerate(_LABELS)), scores

    def _get_xy_samples(self, model_name: str):
        """
        Retrieves cached samples for aesthetic assessment.

        :param model_name: The name of the aesthetic assessment model.
        :type model_name: str
        :return: Cached samples for aesthetic assessment.
        :rtype: Tuple[Tuple[np.ndarray, float, float], Tuple[np.ndarray, float, float]]
        """
        if model_name not in self.cached_samples:
            stacked = np.load(hf_hub_download(
                repo_id=self.repo_id,
                repo_type='model',
                filename=f'{model_name}/samples.npz',
            ))['arr_0']
            x, y = stacked[0], stacked[1]
            self.cached_samples[model_name] = ((x, x.min(), x.max()), (y, y.min(), y.max()))
        return self.cached_samples[model_name]

    def score_to_percentile(self, score: float, model_name: str) -> float:
        """
        Converts an aesthetic score to a percentile rank.

        :param score: The aesthetic score.
        :type score: float
        :param model_name: The name of the aesthetic assessment model to use.
        :type model_name: str
        :return: The percentile rank corresponding to the given score.
        :rtype: float
        """
        (x, x_min, x_max), (y, y_min, y_max) = self._get_xy_samples(model_name)
        idx = np.searchsorted(x, np.clip(score, a_min=x_min, a_max=x_max))
        if idx < x.shape[0] - 1:
            x0, y0 = x[idx], y[idx]
            x1, y1 = x[idx + 1], y[idx + 1]
            if np.isclose(x1, x0):
                return y[idx]
            else:
                return np.clip((score - x0) / (x1 - x0) * (y1 - y0) + y0, a_min=y_min, a_max=y_max)
        else:
            return y[idx]

    @classmethod
    def percentile_to_label(cls, percentile: float, mapping: Optional[Dict[str, float]] = None) -> str:
        """
        Converts a percentile rank to an aesthetic label.

        :param percentile: The percentile rank.
        :type percentile: float
        :param mapping: A dictionary mapping labels to percentile thresholds.
        :type mapping: Optional[Dict[str, float]]
        :return: The aesthetic label corresponding to the given percentile rank.
        :rtype: str
        """
        mapping = mapping or _DEFAULT_LABEL_MAPPING
        for label, threshold in sorted(mapping.items(), key=lambda x: (-x[1], x[0])):
            if percentile >= threshold:
                return label
        else:
            raise ValueError(f'No label for unknown percentile {percentile:.3f}.')  # pragma: no cover

    def get_aesthetic(self, image: ImageTyping, model_name: str, fmt=('label', 'percentile')):
        """
        Analyzes the aesthetic quality of an anime image and returns the results in the specified format.

        :param image: The input anime image.
        :type image: ImageTyping
        :param model_name: The name of the aesthetic assessment model to use.
        :type model_name: str
        :param fmt: The format of the output.
        :type fmt: Tuple[str, ...]
        :return: A dictionary containing the aesthetic assessment results.
        :rtype: Dict[str, float]
        """
        score, confidence = self.get_aesthetic_score(image, model_name)
        percentile = self.score_to_percentile(score, model_name)
        label = self.percentile_to_label(percentile)
        return vreplace(
            v=fmt,
            mapping={
                'label': label,
                'percentile': percentile,
                'score': score,
                'confidence': confidence,
            }
        )

    def clear(self):
        """
        Clears the internal state of the AestheticModel instance.
        """
        self.classifier.clear()
        self.cached_samples.clear()


_MODEL = AestheticModel(_REPO_ID)


def anime_dbaesthetic(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME,
                      fmt=('label', 'percentile')):
    """
    Analyzes the aesthetic quality of an anime image using a pre-trained model.

    :param image: The input anime image.
    :type image: ImageTyping
    :param model_name: The name of the aesthetic assessment model to use. Default is _DEFAULT_MODEL_NAME.
    :type model_name: str
    :param fmt: The format of the output. Default is ('label', 'percentile').
    :type fmt: Tuple[str, ...]
    :return: A dictionary containing the aesthetic assessment results.
    :rtype: Dict[str, float]

    Examples::
        >>> from imgutils.metrics import anime_dbaesthetic
        >>>
        >>> anime_dbaesthetic('masterpiece.jpg')
        ('masterpiece', 0.9831666690063624)
        >>> anime_dbaesthetic('best.jpg')
        ('best', 0.8810615667538594)
        >>> anime_dbaesthetic('great.jpg')
        ('great', 0.8225559148288356)
        >>> anime_dbaesthetic('good.jpg')
        ('good', 0.591020403706702)
        >>> anime_dbaesthetic('normal.jpg')
        ('normal', 0.2888798940585766)
        >>> anime_dbaesthetic('low.jpg')
        ('low', 0.243279223969715)
        >>> anime_dbaesthetic('worst.jpg')
        ('worst', 0.005268185993767627)

        * Custom format

        >>> anime_dbaesthetic('masterpiece.jpg', fmt=('label', 'percentile', 'score'))
        ('masterpiece', 0.9831666690063624, 5.275707557797432)
        >>> anime_dbaesthetic('best.jpg', fmt=('label', 'percentile', 'score'))
        ('best', 0.8810615667538594, 4.7977807857096195)
        >>> anime_dbaesthetic('great.jpg', fmt=('label', 'percentile', 'score'))
        ('great', 0.8225559148288356, 4.56098810210824)
        >>> anime_dbaesthetic('good.jpg', fmt=('label', 'percentile', 'score'))
        ('good', 0.591020403706702, 3.670568235218525)
        >>> anime_dbaesthetic('normal.jpg', fmt=('label', 'percentile', 'score'))
        ('normal', 0.2888798940585766, 2.1677918508648872)
        >>> anime_dbaesthetic('low.jpg', fmt=('label', 'percentile', 'score'))
        ('low', 0.243279223969715, 1.9305131509900093)
        >>> anime_dbaesthetic('worst.jpg', fmt=('label', 'percentile', 'score'))
        ('worst', 0.005268185993767627, 0.6085879728198051)

        * Get confidence

        >>> anime_dbaesthetic('masterpiece.jpg', fmt='confidence')
        {'masterpiece': 0.6834832429885864, 'best': 0.16141420602798462, 'great': 0.05435194447636604, 'good': 0.025083942338824272, 'normal': 0.024000568315386772, 'low': 0.027076328173279762, 'worst': 0.024589713662862778}
        >>> anime_dbaesthetic('best.jpg', fmt='confidence')
        {'masterpiece': 0.3757021427154541, 'best': 0.3451208472251892, 'great': 0.1511985808610916, 'good': 0.04740551486611366, 'normal': 0.02172713913023472, 'low': 0.027498546987771988, 'worst': 0.03134724497795105}
        >>> anime_dbaesthetic('great.jpg', fmt='confidence')
        {'masterpiece': 0.39281174540519714, 'best': 0.22457796335220337, 'great': 0.15563568472862244, 'good': 0.10796019434928894, 'normal': 0.047730278223752975, 'low': 0.0393439345061779, 'worst': 0.031940147280693054}
        >>> anime_dbaesthetic('good.jpg', fmt='confidence')
        {'masterpiece': 0.13832266628742218, 'best': 0.20687267184257507, 'great': 0.2509062886238098, 'good': 0.1644320785999298, 'normal': 0.11332042515277863, 'low': 0.08270663768053055, 'worst': 0.043439216911792755}
        >>> anime_dbaesthetic('normal.jpg', fmt='confidence')
        {'masterpiece': 0.033693961799144745, 'best': 0.03375888615846634, 'great': 0.050045162439346313, 'good': 0.16734018921852112, 'normal': 0.4311050772666931, 'low': 0.23242227733135223, 'worst': 0.05163438618183136}
        >>> anime_dbaesthetic('low.jpg', fmt='confidence')
        {'masterpiece': 0.012833272106945515, 'best': 0.01619996316730976, 'great': 0.03074900433421135, 'good': 0.1396280825138092, 'normal': 0.5038207173347473, 'low': 0.22299200296401978, 'worst': 0.07377689331769943}
        >>> anime_dbaesthetic('worst.jpg', fmt='confidence')
        {'masterpiece': 0.02854202501475811, 'best': 0.026677291840314865, 'great': 0.02838410809636116, 'good': 0.026617199182510376, 'normal': 0.02508518099784851, 'low': 0.06039097160100937, 'worst': 0.8043031692504883}
    """
    return _MODEL.get_aesthetic(image, model_name, fmt)
