"""
Multi-Label TIMM Model Module

This module provides functionality for working with multi-label image classification models
trained with TIMM (PyTorch Image Models) and exported to ONNX format. It includes:

1. The MultiLabelTIMMModel class for loading and making predictions with models hosted on Hugging Face Hub
2. Functions for batch prediction and demo interface creation
3. Support for custom thresholds at both category and tag levels
4. Flexible output formatting options for different use cases

The models are expected to be stored on Hugging Face Hub with specific files:

- model.onnx: The ONNX model file
- selected_tags.csv: CSV file containing tag information and categories
- preprocess.json: JSON configuration for image preprocessing
- thresholds.csv: Optional CSV file with recommended thresholds

This module is designed to work with multi-label classification tasks where images can
belong to multiple categories and have multiple tags within each category.
"""

import io
import json
import os
import warnings
from threading import Lock
from typing import Optional, Literal, Dict, Any, Union, Tuple

import pandas as pd
from hbutils.string import titleize
from hfutils.repository import hf_hub_repo_url
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from ..data import ImageTyping, load_image
from ..preprocess import create_pillow_transforms
from ..utils import open_onnx_model
from ..utils import vreplace, ts_lru_cache

try:
    import gradio as gr
except (ImportError, ModuleNotFoundError):
    gr = None

__all__ = [
    'MultiLabelTIMMModel',
    'multilabel_timm_predict',
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


FMT_UNSET = object()


class MultiLabelTIMMModel:
    """
    A class for working with multi-label image classification models trained with TIMM.

    This class handles loading models from Hugging Face Hub, preprocessing images,
    and making predictions with customizable thresholds.

    :param repo_id: The Hugging Face Hub repository ID containing the model
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token for private repositories
    :type hf_token: Optional[str]
    :param category_names: Optional mapping of category IDs to display names
    :type category_names: Dict[Any, str]
    """

    def __init__(self, repo_id: str, hf_token: Optional[str] = None, category_names: Dict[Any, str] = None):
        """
        Initialize a MultiLabelTIMMModel.

        :param repo_id: The Hugging Face Hub repository ID containing the model
        :type repo_id: str
        :param hf_token: Optional Hugging Face authentication token for private repositories
        :type hf_token: Optional[str]
        :param category_names: Optional mapping of category IDs to display names
        :type category_names: Dict[Any, str]
        """
        self.repo_id = repo_id
        self._model = None
        self._df_tags = None
        self._preprocess = None
        self._default_category_thresholds = None
        self._hf_token = hf_token
        self._lock = Lock()
        self._category_names = category_names or {}
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
        Load the ONNX model from Hugging Face Hub.

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
        Load tag information from the Hugging Face Hub.

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
                self._name_to_categories = {}
                for category in sorted(set(self._df_tags['category'])):
                    if not self._category_names.get(category):
                        self._category_names[category] = f'category_{category}'
                    self._name_to_categories[self._category_names[category]] = category

        return self._df_tags

    def _open_preprocess(self):
        """
        Load preprocessing configuration from the Hugging Face Hub.

        :return: A tuple of validation and test preprocessing transforms
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

    def _open_default_category_thresholds(self):
        """
        Load default category thresholds from the Hugging Face Hub.

        :return: Dictionary mapping category IDs to threshold values
        :rtype: dict
        """
        with self._lock:
            if self._default_category_thresholds is None:
                try:
                    df_category_thresholds = pd.read_csv(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename='thresholds.csv'
                    ))
                except (EntryNotFoundError,):
                    self._default_category_thresholds = {}
                else:
                    self._default_category_thresholds = {}
                    for item in df_category_thresholds.to_dict('records'):
                        if item['category'] not in self._default_category_thresholds:
                            self._default_category_thresholds[item['category']] = item['threshold']

        return self._default_category_thresholds

    def _raw_predict(self, image: ImageTyping, preprocessor: Literal['test', 'val'] = 'test'):
        """
        Make a raw prediction with the model.

        :param image: The input image
        :type image: ImageTyping
        :param preprocessor: Which preprocessor to use ('test' or 'val')
        :type preprocessor: Literal['test', 'val']

        :return: Dictionary of model outputs
        :rtype: dict
        :raises ValueError: If an unknown preprocessor is specified
        """
        image = load_image(image, force_background='white', mode='RGB')
        model = self._open_model()

        val_trans, test_trans = self._open_preprocess()
        if preprocessor == 'test':
            trans = test_trans
        elif preprocessor == 'val':
            trans = val_trans
        else:
            raise ValueError(f'Unknown processor - {preprocessor!r}.')  # pragma: no cover

        input_ = trans(image)[None, ...]
        output_names = [output.name for output in model.get_outputs()]
        output_values = model.run(output_names, {'input': input_})
        return {name: value[0] for name, value in zip(output_names, output_values)}

    def predict(self, image: ImageTyping, preprocessor: Literal['test', 'val'] = 'test',
                thresholds: Union[float, Dict[Any, float]] = None, use_tag_thresholds: bool = True,
                fmt=FMT_UNSET):
        """
        Make a prediction and format the results.

        This method processes an image through the model and applies thresholds to determine
        which tags to include in the results. The output format can be customized using the fmt parameter.

        :param image: The input image
        :type image: ImageTyping
        :param preprocessor: Which preprocessor to use ('test' or 'val')
        :type preprocessor: Literal['test', 'val']
        :param thresholds: Threshold values for tag confidence. Can be a single float applied to all categories
                          or a dictionary mapping category IDs or names to threshold values
        :type thresholds: Union[float, Dict[Any, float]]
        :param use_tag_thresholds: Whether to use tag-level thresholds if available
        :type use_tag_thresholds: bool
        :param fmt: Output format specification. Can be a tuple of category names to include,
                   or FMT_UNSET to use all categories
        :type fmt: Any

        :return: Formatted prediction results according to the fmt parameter
        :rtype: Any

        .. note::
            The fmt argument can include the following keys:

            - Category names: dicts containing category-specific tags and their confidences
            - ``tag``: a dict containing all tags across categories and their confidences
            - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
            - ``logits``: a 1-dim logits result of image.
            - ``prediction``: a 1-dim prediction result of image

            You can extract specific category predictions or all tags based on your needs.
        """
        df_tags = self._open_tags()
        values = self._raw_predict(image, preprocessor=preprocessor)
        prediction = values['prediction']
        tags = {}

        if fmt is FMT_UNSET:
            fmt = tuple(self._category_names[category] for category in sorted(set(df_tags['category'].tolist())))

        default_category_thresholds = self._open_default_category_thresholds()
        if 'best_threshold' in self._df_tags:
            default_tag_thresholds = self._df_tags['best_threshold']
        else:
            default_tag_thresholds = None
        for category in sorted(set(df_tags['category'].tolist())):
            mask = df_tags['category'] == category
            tag_names = df_tags['name'][mask]
            category_pred = prediction[mask]

            if isinstance(thresholds, float):
                category_threshold = thresholds
            elif isinstance(thresholds, dict) and \
                    (category in thresholds or self._category_names[category] in thresholds):
                if category in thresholds:
                    category_threshold = thresholds[category]
                elif self._category_names[category] in thresholds:
                    category_threshold = thresholds[self._category_names[category]]
                else:
                    assert False, 'Should not reach this line'  # pragma: no cover
            elif use_tag_thresholds and default_tag_thresholds is not None:
                category_threshold = default_tag_thresholds[mask]
            else:
                if use_tag_thresholds:
                    warnings.warn(f'Tag thresholds not supported in model {self.repo_id!r}.')
                if category in default_category_thresholds:
                    category_threshold = default_category_thresholds[category]
                else:
                    category_threshold = 0.4

            mask = category_pred >= category_threshold
            tag_names = tag_names[mask].tolist()
            category_pred = category_pred[mask].tolist()
            cate_tags = dict(sorted(zip(tag_names, category_pred), key=lambda x: (-x[1], x[0])))
            values[self._category_names[category]] = cate_tags
            tags.update(cate_tags)

        values['tag'] = tags
        return vreplace(fmt, values)

    def make_ui(self, default_thresholds: Union[float, Dict[Any, float]] = None,
                default_use_tag_thresholds: bool = True):
        """
        Create a Gradio UI for the model.

        :param default_thresholds: Default threshold values to use in the UI
        :type default_thresholds: Union[float, Dict[Any, float]]
        :param default_use_tag_thresholds: Whether to use tag-level thresholds by default
        :type default_use_tag_thresholds: bool

        :return: None
        :raises EnvironmentError: If Gradio is not installed
        """
        _check_gradio_env()
        df_tags = self._open_tags()
        default_category_thresholds = self._open_default_category_thresholds()
        allow_use_tag_thresholds = 'best_threshold' in self._df_tags

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    gr_input_image = gr.Image(type='pil', label='Original Image')
                with gr.Row(visible=allow_use_tag_thresholds):
                    gr_use_tag_thresholds = gr.Checkbox(
                        value=allow_use_tag_thresholds and default_use_tag_thresholds,
                        label='Use Tag-Level Thresholds',
                        interactive=allow_use_tag_thresholds,
                        visible=allow_use_tag_thresholds,
                    )
                    gr.HTML(
                        value="<div style='font-size: 0.8em; color: var(--color-text-secondary); margin-top: 0.3em;'>"
                              "<b>Note:</b> Category thresholds will be ignored when tag-level thresholds enabled!!!</div>",
                        visible=allow_use_tag_thresholds
                    )
                with gr.Row():
                    gr_thresholds = []
                    for category in sorted(set(df_tags['category'].tolist())):
                        if isinstance(default_thresholds, float):
                            category_threshold = default_thresholds
                        elif isinstance(default_thresholds, dict) and \
                                (category in default_thresholds or self._category_names[
                                    category] in default_thresholds):
                            if category in default_thresholds:
                                category_threshold = default_thresholds[category]
                            elif self._category_names[category] in default_thresholds:
                                category_threshold = default_thresholds[self._category_names[category]]
                            else:
                                assert False, 'Should not reach this line'  # pragma: no cover
                        elif category in default_category_thresholds:
                            category_threshold = default_category_thresholds[category]
                        else:
                            category_threshold = 0.4

                        gr_cate_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=category_threshold,
                            step=0.001,
                            label=f'Threshold for {titleize(self._category_names[category])}',
                        )
                        gr_thresholds.append(gr_cate_threshold)

                with gr.Row():
                    gr_submit = gr.Button(value='Submit', variant='primary')

            with gr.Column():
                with gr.Tabs():
                    gr_preds = []
                    for category in sorted(set(df_tags['category'].tolist())):
                        with gr.Tab(f'{titleize(self._category_names[category])}'):
                            gr_cate_label = gr.Label(f'{titleize(self._category_names[category])} Prediction')
                            gr_preds.append(gr_cate_label)

                    with gr.Tab('Text Output'):
                        gr_text_output = gr.TextArea(label="Output (string)", lines=15)

            def _fn_submit(image, _use_tag_thresholds, *thresholds):
                if _use_tag_thresholds:
                    _ths = None
                else:
                    _ths = {
                        category: cate_ths
                        for category, cate_ths in zip(sorted(set(df_tags['category'].tolist())), thresholds)
                    }

                fmt = tuple(self._category_names[category] for category in sorted(set(df_tags['category'].tolist())))
                res = self.predict(
                    image=image,
                    thresholds=_ths,
                    use_tag_thresholds=_use_tag_thresholds,
                    fmt=fmt,
                )
                with io.StringIO() as sf:
                    for category, res_item in zip(sorted(set(df_tags['category'].tolist())), res):
                        print(f'# {self._category_names[category]} (#{category}):', file=sf)
                        print(', '.join(res_item.keys()), file=sf)
                        print('', file=sf)

                    return sf.getvalue(), *res

            gr_submit.click(
                fn=_fn_submit,
                inputs=[gr_input_image, gr_use_tag_thresholds, *gr_thresholds],
                outputs=[gr_text_output, *gr_preds]
            )

    def launch_demo(self, default_thresholds: Union[float, Dict[Any, float]] = None,
                    default_use_tag_thresholds: bool = True,
                    server_name: Optional[str] = None, server_port: Optional[int] = None, **kwargs):
        """
        Launch a Gradio demo for the model.

        :param default_thresholds: Default threshold values to use in the demo
        :type default_thresholds: Union[float, Dict[Any, float]]
        :param default_use_tag_thresholds: Whether to use tag-level thresholds by default
        :type default_use_tag_thresholds: bool
        :param server_name: Server name for the Gradio app
        :type server_name: Optional[str]
        :param server_port: Server port for the Gradio app
        :type server_port: Optional[int]
        :param kwargs: Additional keyword arguments to pass to gr.launch()
        :type kwargs: Any

        :return: None
        :raises EnvironmentError: If Gradio is not installed
        """
        _check_gradio_env()
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    repo_url = hf_hub_repo_url(repo_id=self.repo_id, repo_type='model')
                    gr.HTML(f'<h2 style="text-align: center;">Tagger Demo For {self.repo_id}</h2>')
                    gr.Markdown(f'This is the quick demo for tagger model [{self.repo_id}]({repo_url}). '
                                f'Powered by `dghs-imgutils`\'s quick demo module.')

            with gr.Row():
                self.make_ui(
                    default_thresholds=default_thresholds,
                    default_use_tag_thresholds=default_use_tag_thresholds,
                )

        demo.launch(
            server_name=server_name,
            server_port=server_port,
            **kwargs,
        )


@ts_lru_cache()
def _open_models_for_repo_id(repo_id: str, category_names: Optional[Tuple[Tuple[Any, str], ...]] = None,
                             hf_token: Optional[str] = None) \
        -> MultiLabelTIMMModel:
    """
    Open and cache a MultiLabelTIMMModel for a given repository ID.

    :param repo_id: The Hugging Face Hub repository ID
    :type repo_id: str
    :param category_names: Optional tuple of (category_id, name) pairs for category naming
    :type category_names: Optional[Tuple[Tuple[Any, str], ...]]
    :param hf_token: Optional Hugging Face authentication token
    :type hf_token: Optional[str]

    :return: A cached MultiLabelTIMMModel instance
    :rtype: MultiLabelTIMMModel
    """
    return MultiLabelTIMMModel(
        repo_id=repo_id,
        hf_token=hf_token,
        category_names=dict(category_names or []),
    )


def multilabel_timm_predict(image: ImageTyping, repo_id: str, category_names: Dict[Any, str] = None,
                            preprocessor: Literal['test', 'val'] = 'test',
                            thresholds: Union[float, Dict[Any, float]] = None, use_tag_thresholds: bool = True,
                            fmt=FMT_UNSET, hf_token: Optional[str] = None):
    """
    Make predictions using a multi-label TIMM model.

    This function provides a convenient interface for making predictions with models
    hosted on Hugging Face Hub without directly instantiating a MultiLabelTIMMModel.

    :param image: The input image
    :type image: ImageTyping
    :param repo_id: The Hugging Face Hub repository ID containing the model
    :type repo_id: str
    :param category_names: Optional mapping of category IDs to display names
    :type category_names: Dict[Any, str]
    :param preprocessor: Which preprocessor to use ('test' or 'val')
    :type preprocessor: Literal['test', 'val']
    :param thresholds: Threshold values for tag confidence. Can be a single float applied to all categories
                      or a dictionary mapping category IDs or names to threshold values
    :type thresholds: Union[float, Dict[Any, float]]
    :param use_tag_thresholds: Whether to use tag-level thresholds if available
    :type use_tag_thresholds: bool
    :param fmt: Output format specification. Can be a tuple of category names to include,
               or FMT_UNSET to use all categories
    :type fmt: Any
    :param hf_token: Optional Hugging Face authentication token for private repositories
    :type hf_token: Optional[str]

    :return: Formatted prediction results according to the fmt parameter
    :rtype: Any

    .. note::
        The fmt argument can include the following keys:

        - Category names: dicts containing category-specific tags and their confidences
        - ``tag``: a dict containing all tags across categories and their confidences
        - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
        - ``logits``: a 1-dim logits result of image.
        - ``prediction``: a 1-dim prediction result of image

        You can extract specific category predictions or all tags based on your needs.

        Example:

        >>> # Get all categories and tags
        >>> result = multilabel_timm_predict('image.jpg', 'username/model-name')
        >>>
        >>> # Get only specific categories
        >>> result = multilabel_timm_predict('image.jpg', 'username/model-name',
        ...                                  fmt=('general', 'character'))
        >>>
        >>> # Get just the raw prediction values
        >>> prediction = multilabel_timm_predict('image.jpg', 'username/model-name',
        ...                                     fmt='prediction')
    """
    model = _open_models_for_repo_id(
        repo_id=repo_id,
        category_names=tuple((key, value) for key, value in sorted((category_names or {}).items())),
        hf_token=hf_token,
    )
    return model.predict(
        image=image,
        preprocessor=preprocessor,
        thresholds=thresholds,
        use_tag_thresholds=use_tag_thresholds,
        fmt=fmt,
    )
