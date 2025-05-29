"""
TIMM-based Image Classification Module

This module provides functionality for using pre-trained TIMM (PyTorch Image Models) models
for image classification tasks. It includes capabilities for:

- Loading TIMM models from Hugging Face repositories
- Processing and classifying images
- Creating interactive web demos with Gradio
- Retrieving and formatting prediction results

The module is designed to make it easy to use pre-trained image classification models
with minimal setup, supporting both programmatic use and interactive demos.
"""

import json
import os
import re
from threading import Lock
from typing import Optional, Literal

import numpy as np
import pandas as pd
from hfutils.repository import hf_hub_repo_url
from huggingface_hub import hf_hub_download

from ..data import ImageTyping, load_image
from ..preprocess import create_pillow_transforms
from ..utils import open_onnx_model, vnames
from ..utils import vreplace, ts_lru_cache

try:
    import gradio as gr
except (ImportError, ModuleNotFoundError):
    gr = None

__all__ = [
    'ClassifyTIMMModel',
    'classify_timm_predict',
]


def _check_gradio_env():
    """
    Verify that Gradio library is properly installed and available.

    This function checks if the Gradio package is accessible for creating
    web-based demos. If Gradio is not found, it provides instructions for installation.

    :raises EnvironmentError: If Gradio package is not installed in the environment.
    """
    if gr is None:
        raise EnvironmentError(f'Gradio required for launching webui-based demo.\n'
                               f'Please install it with `pip install dghs-imgutils[demo]`.')


class ClassifyTIMMModel:
    """
    A class for handling TIMM-based image classification models.

    This class provides functionality to load models from Hugging Face repositories,
    perform predictions, and create interactive demos.

    :param repo_id: The Hugging Face repository ID for the model
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token
    :type hf_token: Optional[str]
    """

    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        """
        Initialize a ClassifyTIMMModel instance.

        :param repo_id: The Hugging Face repository ID for the model
        :type repo_id: str
        :param hf_token: Optional Hugging Face authentication token
        :type hf_token: Optional[str]
        """
        self.repo_id = repo_id
        self._model = None
        self._df_tags = None
        self._preprocess = None
        self._hf_token = hf_token
        self._lock = Lock()
        self._name_to_categories = None

    def _get_hf_token(self) -> Optional[str]:
        """
        Retrieve the Hugging Face authentication token.

        Checks both instance variable and environment for token presence.

        :return: Authentication token if available
        :rtype: Optional[str]
        """
        return self._hf_token or os.environ.get('HF_TOKEN')

    def _open_model(self):
        """
        Load the ONNX model from Hugging Face repository.

        This method ensures thread-safe loading of the model and caches it for reuse.

        :return: The loaded ONNX model
        :rtype: object
        """
        with self._lock:
            if self._model is None:
                self._model = open_onnx_model(hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename='model.onnx',
                    token=self._get_hf_token(),
                ))

        return self._model

    def _open_tags(self):
        """
        Load the tags data from Hugging Face repository.

        This method ensures thread-safe loading of the tags and caches them for reuse.

        :return: DataFrame containing tag information
        :rtype: pandas.DataFrame
        """
        with self._lock:
            if self._df_tags is None:
                self._df_tags = pd.read_csv(hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename='selected_tags.csv',
                    token=self._get_hf_token(),
                ))

        return self._df_tags

    def _open_preprocess(self):
        """
        Load the preprocessing configuration from Hugging Face repository.

        This method ensures thread-safe loading of the preprocessing configuration
        and caches it for reuse.

        :return: A tuple of validation and test transform functions
        :rtype: tuple
        """
        with self._lock:
            if self._preprocess is None:
                with open(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename='preprocess.json'
                ), 'r') as f:
                    data_ = json.load(f)
                    test_trans = create_pillow_transforms(data_['test'])
                    val_trans = create_pillow_transforms(data_['val'])
                    self._preprocess = val_trans, test_trans

        return self._preprocess

    def _raw_predict(self, image: ImageTyping, preprocessor: Literal['test', 'val'] = 'test'):
        """
        Perform raw prediction on an image.

        This method handles the preprocessing and model inference steps.

        :param image: The input image to classify
        :type image: ImageTyping
        :param preprocessor: Which preprocessor to use ('test' or 'val')
        :type preprocessor: Literal['test', 'val']

        :return: Dictionary of model outputs
        :rtype: dict
        :raises ValueError: If an invalid preprocessor type is specified
        """
        image = load_image(image, force_background='white', mode='RGB')
        model = self._open_model()

        val_trans, test_trans = self._open_preprocess()
        if preprocessor == 'test':
            trans = test_trans
        elif preprocessor == 'val':
            trans = val_trans
        else:
            raise ValueError(
                f'Unknown processor, "test" or "val" expected but {preprocessor!r} found.')  # pragma: no cover

        input_ = trans(image)[None, ...]
        output_names = [output.name for output in model.get_outputs()]
        output_values = model.run(output_names, {'input': input_})
        return {name: value[0] for name, value in zip(output_names, output_values)}

    def predict(self, image: ImageTyping, preprocessor: Literal['test', 'val'] = 'test', fmt='scores-top5'):
        """
        Predict classification results for an image.

        This method processes the image, runs inference, and formats the results
        according to the specified format.

        :param image: The input image to classify
        :type image: ImageTyping
        :param preprocessor: Which preprocessor to use ('test' or 'val')
        :type preprocessor: Literal['test', 'val']
        :param fmt: Output format specification, e.g., 'scores-top5'

        :return: Formatted prediction results
        :rtype: dict or other type depending on fmt

        .. note::
            The ``fmt`` argument can include the following keys:

            - ``scores``: dicts containing all the prediction scores of all the classes, may be a very big dict
            - ``scores-top<k>``: dict containing top-k classes and their scores, e.g. ``scores-top5`` means top-5 classes
            - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
            - ``logits``: a 1-dim logits result of image
            - ``prediction``: a 1-dim prediction result of image

            You can extract specific category predictions or all tags based on your needs.

        For more details see documentation of :func:`classify_timm_predict`.
        """
        df_tags = self._open_tags()
        values = self._raw_predict(image, preprocessor=preprocessor)
        prediction = values['prediction']

        for vname in vnames(fmt, str_only=True):
            matching = re.fullmatch(r'^scores(-top(?P<topk>\d+))?$', vname)
            if matching:
                topk = int(matching.group('topk')) if matching.group('topk') else None
                order = np.argsort(-prediction)
                if topk is not None:
                    order = order[:topk]
                pred = prediction[order].tolist()
                labs = df_tags['name'][order].tolist()
                values[vname] = dict(zip(labs, pred))

        return vreplace(fmt, values)

    def make_ui(self):
        """
        Create a Gradio UI component for the model.

        This method builds a user interface for interactive image classification
        that can be embedded in a larger Gradio application.

        :raises EnvironmentError: If Gradio is not installed
        """
        _check_gradio_env()

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    gr_input_image = gr.Image(type='pil', label='Original Image')
                with gr.Row():
                    gr_topk = gr.Slider(minimum=1, maximum=30, value=5, step=1, label='Top-K')
                with gr.Row():
                    gr_submit = gr.Button(value='Submit', variant='primary')

            with gr.Column():
                gr_pred = gr.Label(label='Prediction')

            def _fn_submit(image, topk):
                return self.predict(
                    image=image,
                    fmt=f'scores-top{topk}',
                )

            gr_submit.click(
                fn=_fn_submit,
                inputs=[gr_input_image, gr_topk],
                outputs=[gr_pred]
            )

    def launch_demo(self, server_name: Optional[str] = None, server_port: Optional[int] = None, **kwargs):
        """
        Launch a standalone Gradio demo for the model.

        This method creates and launches a complete Gradio web application
        for interactive image classification.

        :param server_name: Server name for the Gradio app
        :type server_name: Optional[str]
        :param server_port: Server port for the Gradio app
        :type server_port: Optional[int]
        :param kwargs: Additional keyword arguments to pass to gr.Blocks.launch()

        :raises EnvironmentError: If Gradio is not installed
        """
        _check_gradio_env()
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    repo_url = hf_hub_repo_url(repo_id=self.repo_id, repo_type='model')
                    gr.HTML(f'<h2 style="text-align: center;">TIMM-based Classifier Demo For {self.repo_id}</h2>')
                    gr.Markdown(f'This is the quick demo for tagger model [{self.repo_id}]({repo_url}). '
                                f'Powered by `dghs-imgutils`\'s quick demo module.')

            with gr.Row():
                self.make_ui()

        demo.launch(
            server_name=server_name,
            server_port=server_port,
            **kwargs,
        )


@ts_lru_cache()
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) -> ClassifyTIMMModel:
    """
    Open and cache a ClassifyTIMMModel for a given repository ID.

    This function uses time-sensitive LRU caching to avoid repeatedly loading
    the same model.

    :param repo_id: The Hugging Face repository ID for the model
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token
    :type hf_token: Optional[str]

    :return: A ClassifyTIMMModel instance
    :rtype: ClassifyTIMMModel
    """
    return ClassifyTIMMModel(
        repo_id=repo_id,
        hf_token=hf_token,
    )


def classify_timm_predict(image: ImageTyping, repo_id: str, preprocessor: Literal['test', 'val'] = 'test',
                          fmt='scores-top5', hf_token: Optional[str] = None):
    """
    Perform image classification using a TIMM model from a Hugging Face repository.

    This is a convenience function that handles model loading and prediction in one call.

    :param image: The input image to classify
    :type image: ImageTyping
    :param repo_id: The Hugging Face repository ID for the model
    :type repo_id: str
    :param preprocessor: Which preprocessor to use ('test' or 'val')
    :type preprocessor: Literal['test', 'val']
    :param fmt: Output format specification, e.g., 'scores-top5'
    :param hf_token: Optional Hugging Face authentication token
    :type hf_token: Optional[str]

    :return: Formatted prediction results
    :rtype: dict or other type depending on fmt

    Example:
        Here are some images for example

        .. image:: classify_timm_demo.plot.py.svg
           :align: center

        >>> from imgutils.generic import classify_timm_predict
        >>>
        >>> classify_timm_predict(
        ...     'classify_timm/img1.jpg',
        ...     repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ... )
        {'jia_redian_ruzi_ruzi': 0.9890832304954529, 'siya_ho': 0.005189628805965185, 'bai_qi-qsr': 0.0015026535838842392, 'kkuem': 0.0012714712647721171, 'teddy_(khanshin)': 0.00035598213435150683}
        >>>
        >>> classify_timm_predict(
        ...     'classify_timm/img2.jpg',
        ...     repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ... )
        {'monori_rogue': 0.6921895742416382, 'stanley_lau': 0.2040979117155075, 'neoartcore': 0.03475344926118851, 'ayya_sap': 0.005350438412278891, 'goomrrat': 0.004616163671016693}
        >>>
        >>> classify_timm_predict(
        ...     'classify_timm/img3.jpg',
        ...     repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ... )
        {'shexyo': 0.9998241066932678, 'oroborus': 0.0001537767384434119, 'jeneral': 7.268482477229554e-06, 'free_style_(yohan1754)': 3.4537688406999223e-06, 'kakeku': 2.5340586944366805e-06}
        >>>
        >>> classify_timm_predict(
        ...     'classify_timm/img4.jpg',
        ...     repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ... )
        {'z.taiga': 0.9999995231628418, 'tina_(tinafya)': 1.2290533391023928e-07, 'arind_yudha': 6.17258208990279e-08, 'chixiao': 4.949555076905199e-08, 'zerotwenty_(020)': 4.218352955831506e-08}
        >>>
        >>> classify_timm_predict(
        ...     'classify_timm/img5.jpg',
        ...     repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ... )
        {'spam_(spamham4506)': 0.9999998807907104, 'falken_(yutozin)': 4.501828954062148e-08, 'yuki_(asayuki101)': 3.285677863118508e-08, 'danbal': 5.452678752959628e-09, 'buri_(retty9349)': 3.757136379789472e-09}
        >>>
        >>> classify_timm_predict(
        ...     'classify_timm/img6.jpg',
        ...     repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ... )
        {'mashuu_(neko_no_oyashiro)': 1.0, 'minaba_hideo': 4.543745646401476e-08, 'simosi': 6.499865978781827e-09, 'maoh_yueer': 4.302619149854081e-09, '7nite': 3.6548184478846224e-09}

    .. note::
        The ``fmt`` argument can include the following keys:

        - ``scores``: dicts containing all the prediction scores of all the classes, may be a very big dict
        - ``scores-top<k>``: dict containing top-k classes and their scores, e.g. ``scores-top5`` means top-5 classes
        - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
        - ``logits``: a 1-dim logits result of image
        - ``prediction``: a 1-dim prediction result of image

        You can extract specific category predictions or all tags based on your needs.

        >>> from imgutils.generic import classify_timm_predict
        >>>
        >>> classify_timm_predict(
        ...     'classify_timm/img1.jpg',
        ...     repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls'
        ... )
        {'jia_redian_ruzi_ruzi': 0.9890832304954529, 'siya_ho': 0.005189628805965185, 'bai_qi-qsr': 0.0015026535838842392, 'kkuem': 0.0012714712647721171, 'teddy_(khanshin)': 0.00035598213435150683}
        >>> embedding = classify_timm_predict(
        ...     'classify_timm/img1.jpg',
        ...     repo_id='animetimm/swinv2_base_window8_256.dbv4a-fullxx-cls',
        ...     fmt='embedding'
        ... )
        >>> embedding.shape, embedding.dtype
        ((1024,), dtype('float32'))
    """
    model = _open_models_for_repo_id(
        repo_id=repo_id,
        hf_token=hf_token,
    )
    return model.predict(
        image=image,
        preprocessor=preprocessor,
        fmt=fmt,
    )
