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
- categories.json: Category ID and name mapping json file.

This module is designed to work with multi-label classification tasks where images can
belong to multiple categories and have multiple tags within each category.
"""

import io
import json
import os
import warnings
from threading import Lock
from typing import Optional, Literal, Dict, Any, Union

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
    """

    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        """
        Initialize a MultiLabelTIMMModel.

        :param repo_id: The Hugging Face Hub repository ID containing the model
        :type repo_id: str
        :param hf_token: Optional Hugging Face authentication token for private repositories
        :type hf_token: Optional[str]
        """
        self.repo_id = repo_id
        self._model = None
        self._df_tags = None
        self._preprocess = None
        self._default_category_thresholds = None
        self._hf_token = hf_token
        self._lock = Lock()
        self._category_names = {}
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
                ), keep_default_na=False)

                with open(hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='model',
                        filename='categories.json',
                        token=self._get_hf_token(),
                ), 'r') as f:
                    d_category_names = {cate_item['category']: cate_item['name'] for cate_item in json.load(f)}
                    self._name_to_categories = {}
                    for category in sorted(set(self._df_tags['category'])):
                        self._category_names[category] = d_category_names[category]
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
                    ), keep_default_na=False)
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
            raise ValueError(
                f'Unknown processor, "test" or "val" expected but {preprocessor!r} found.')  # pragma: no cover

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
            - ``logits``: a 1-dim logits result of image
            - ``prediction``: a 1-dim prediction result of image

            You can extract specific category predictions or all tags based on your needs.

        For more details see documentation of :func:`multilabel_timm_predict`.
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
                        print(f'# {self._category_names[category]} (#{category})', file=sf)
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
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) \
        -> MultiLabelTIMMModel:
    """
    Open and cache a MultiLabelTIMMModel for a given repository ID.

    :param repo_id: The Hugging Face Hub repository ID
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token
    :type hf_token: Optional[str]

    :return: A cached MultiLabelTIMMModel instance
    :rtype: MultiLabelTIMMModel
    """
    return MultiLabelTIMMModel(
        repo_id=repo_id,
        hf_token=hf_token,
    )


def multilabel_timm_predict(image: ImageTyping, repo_id: str,
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

    Example:
        Here are some images for example

        .. image:: multilabel_demo.plot.py.svg
           :align: center

        >>> from imgutils.generic import multilabel_timm_predict
        >>>
        >>> general, character, rating = multilabel_timm_predict(
        ...     'skadi.jpg',
        ...     repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
        ... )
        >>> general
        {'1girl': 0.9963783025741577, 'long_hair': 0.9685494899749756, 'solo': 0.9548443555831909, 'navel': 0.9415484666824341, 'breasts': 0.9369214177131653, 'red_eyes': 0.9019639492034912, 'shirt': 0.873087465763092, 'outdoors': 0.866461992263794, 'crop_top': 0.862577497959137, 'midriff': 0.8544420003890991, 'sportswear': 0.849435567855835, 'shorts': 0.8209151029586792, 'short_sleeves': 0.817188560962677, 'holding': 0.811793327331543, 'very_long_hair': 0.8082301616668701, 'gloves': 0.7840366363525391, 'black_gloves': 0.7765430808067322, 'thighs': 0.7542579770088196, 'looking_at_viewer': 0.7331588268280029, 'day': 0.7203925251960754, 'hair_between_eyes': 0.7121687531471252, 'large_breasts': 0.6990523338317871, 'baseball_bat': 0.6809443831443787, 'grey_hair': 0.6790007948875427, 'sky': 0.6716539263725281, 'stomach': 0.6698249578475952, 'sweat': 0.6454322934150696, 'black_shirt': 0.6270318031311035, 'cowboy_shot': 0.6216483116149902, 'blue_sky': 0.5898874998092651, 'black_shorts': 0.5445142984390259, 'holding_baseball_bat': 0.5013713836669922, 'white_hair': 0.4999670684337616, 'blush': 0.4860053062438965, 'cloud': 0.474183052778244, 'standing': 0.4724341332912445, 'thigh_gap': 0.4330931305885315, 'short_shorts': 0.39793258905410767, 'parted_lips': 0.36694538593292236, 'crop_top_overhang': 0.3321989178657532, 'official_alternate_costume': 0.3157039284706116, 'blurry': 0.24181532859802246, 'groin': 0.21906554698944092, 'ass_visible_through_thighs': 0.2188207507133484, 'cropped_shirt': 0.18700966238975525, 'taut_shirt': 0.08612403273582458, 'taut_clothes': 0.0701744556427002}
        >>> character
        {'skadi_(arknights)': 0.9796262979507446}
        >>> rating
        {'sensitive': 0.9580697417259216}
        >>>
        >>> general, character, rating = multilabel_timm_predict(
        ...     'hutao.jpg',
        ...     repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
        ... )
        >>> general
        {'1girl': 0.988956093788147, 'twintails': 0.9650213718414307, 'ghost': 0.940951943397522, 'tongue_out': 0.9330000877380371, 'tongue': 0.9267600774765015, 'skirt': 0.9194451570510864, 'symbol-shaped_pupils': 0.9103127717971802, 'brown_hair': 0.9067947268486023, 'long_hair': 0.8872615098953247, 'red_eyes': 0.8631541728973389, 'looking_at_viewer': 0.8235997557640076, 'solo': 0.8214132785797119, 'long_sleeves': 0.7965610027313232, 'bag': 0.7958617210388184, 'jacket': 0.7932659387588501, 'flower-shaped_pupils': 0.7630170583724976, 'shirt': 0.7500981092453003, 'hair_ornament': 0.738053023815155, 'flower': 0.7321316599845886, 'plaid_skirt': 0.7173646688461304, 'white_shirt': 0.6631225347518921, 'pleated_skirt': 0.6344470977783203, 'hair_flower': 0.6293849945068359, 'nail_polish': 0.6136130094528198, 'multicolored_hair': 0.5703858733177185, 'blush': 0.5195141434669495, 'plaid_clothes': 0.503984808921814, 'gradient_hair': 0.49658203125, 'alternate_costume': 0.4947473704814911, ':p': 0.493851900100708, 'hair_between_eyes': 0.484821081161499, 'smile': 0.4778161942958832, 'black_nails': 0.4747253358364105, 'collared_shirt': 0.46951043605804443, 'outdoors': 0.46920245885849, 'holding': 0.45227500796318054, 'school_uniform': 0.4197554290294647, 'very_long_hair': 0.41959843039512634, 'miniskirt': 0.3916422426700592, 'cowboy_shot': 0.38207799196243286, 'blue_jacket': 0.3614964485168457, 'sleeves_past_wrists': 0.3611966073513031, 'backpack': 0.32487112283706665, 'colored_tips': 0.314140260219574, 'sidelocks': 0.3062695264816284, 'black_jacket': 0.299169659614563, 'standing': 0.29005059599876404, 'charm_(object)': 0.22183549404144287, 'multiple_rings': 0.2172674536705017, 'open_jacket': 0.2046721875667572, 'ring': 0.18625634908676147, 'brown_skirt': 0.18045437335968018, 'contemporary': 0.13890522718429565}
        >>> character
        {'hu_tao_(genshin_impact)': 0.9779937267303467, 'boo_tao_(genshin_impact)': 0.8973554372787476}
        >>> rating
        {'general': 0.6215817332267761, 'sensitive': 0.3872501254081726}

    .. note::
        For different models, the default format is different. That depends on the categories that model supported.

        For example, for model `animetimm/mobilenetv3_large_150d.dbv4-full-witha <https://huggingface.co/animetimm/mobilenetv3_large_150d.dbv4-full-witha>`_

        >>> from imgutils.generic import multilabel_timm_predict
        >>>
        >>> general, artist, character, rating = multilabel_timm_predict(
        ...     'skadi.jpg',
        ...     repo_id='animetimm/mobilenetv3_large_150d.dbv4-full-witha',
        ... )
        >>> general
        {'1girl': 0.9938606023788452, 'long_hair': 0.9691187143325806, 'red_eyes': 0.9463587403297424, 'solo': 0.944723904132843, 'navel': 0.9439248442649841, 'breasts': 0.9335891008377075, 'sportswear': 0.8865424394607544, 'shorts': 0.8601726293563843, 'very_long_hair': 0.8445472717285156, 'outdoors': 0.83197021484375, 'midriff': 0.8274217247962952, 'shirt': 0.8188955783843994, 'short_sleeves': 0.8183804750442505, 'crop_top': 0.8089936971664429, 'gloves': 0.8038264513015747, 'black_gloves': 0.7703496813774109, 'thighs': 0.7689077854156494, 'holding': 0.768336832523346, 'looking_at_viewer': 0.739115834236145, 'large_breasts': 0.7282243967056274, 'sky': 0.6852632761001587, 'hair_between_eyes': 0.6799711585044861, 'stomach': 0.6694454550743103, 'baseball_bat': 0.6693665385246277, 'black_shorts': 0.6493985652923584, 'day': 0.6425715684890747, 'cowboy_shot': 0.6186742186546326, 'black_shirt': 0.5906491279602051, 'holding_baseball_bat': 0.5860112905502319, 'sweat': 0.5825777649879456, 'cloud': 0.5549533367156982, 'blue_sky': 0.5523971915245056, 'white_hair': 0.5324308276176453, 'grey_hair': 0.52657151222229, 'short_shorts': 0.4896492063999176, 'standing': 0.45526784658432007, 'parted_lips': 0.4306206703186035, 'blush': 0.4149143397808075, 'thigh_gap': 0.4124316871166229, 'ass_visible_through_thighs': 0.34030789136886597, 'artist_name': 0.2679593563079834, 'groin': 0.2652612328529358, 'blurry': 0.2548949122428894, 'baseball': 0.24870169162750244, 'crop_top_overhang': 0.2240566909313202, 'stretching': 0.2012709677219391, 'cropped_shirt': 0.19828352332115173, 'official_alternate_costume': 0.1960265338420868, 'toned': 0.13941210508346558, 'exercising': 0.11270403861999512, 'lens_flare': 0.10835999250411987, 'taut_clothes': 0.08783495426177979, 'taut_shirt': 0.08448180556297302, 'linea_alba': 0.06583884358406067}
        >>> artist
        {}
        >>> character
        {'skadi_(arknights)': 0.8951651453971863}
        >>> rating
        {'sensitive': 0.9492285847663879}

        Its default fmt is ``('general', 'artist', 'character', 'rating')``.

        But you can easily get those information you need with a more controllable way with ``fmt``. See the next note.

    .. note::
        The ``fmt`` argument can include the following keys:

        - Category names: dicts containing category-specific tags and their confidences
        - ``tag``: a dict containing all tags across categories and their confidences
        - ``embedding``: a 1-dim embedding of image, recommended for index building after L2 normalization
        - ``logits``: a 1-dim logits result of image
        - ``prediction``: a 1-dim prediction result of image

        You can extract specific category predictions or all tags based on your needs.

        >>> from imgutils.generic import multilabel_timm_predict
        >>>
        >>> # default usage
        >>> general, character, rating = multilabel_timm_predict(
        ...     'skadi.jpg',
        ...     repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
        ... )
        >>> general
        {'1girl': 0.9963783025741577, 'long_hair': 0.9685494899749756, 'solo': 0.9548443555831909, 'navel': 0.9415484666824341, 'breasts': 0.9369214177131653, 'red_eyes': 0.9019639492034912, 'shirt': 0.873087465763092, 'outdoors': 0.866461992263794, 'crop_top': 0.862577497959137, 'midriff': 0.8544420003890991, 'sportswear': 0.849435567855835, 'shorts': 0.8209151029586792, 'short_sleeves': 0.817188560962677, 'holding': 0.811793327331543, 'very_long_hair': 0.8082301616668701, 'gloves': 0.7840366363525391, 'black_gloves': 0.7765430808067322, 'thighs': 0.7542579770088196, 'looking_at_viewer': 0.7331588268280029, 'day': 0.7203925251960754, 'hair_between_eyes': 0.7121687531471252, 'large_breasts': 0.6990523338317871, 'baseball_bat': 0.6809443831443787, 'grey_hair': 0.6790007948875427, 'sky': 0.6716539263725281, 'stomach': 0.6698249578475952, 'sweat': 0.6454322934150696, 'black_shirt': 0.6270318031311035, 'cowboy_shot': 0.6216483116149902, 'blue_sky': 0.5898874998092651, 'black_shorts': 0.5445142984390259, 'holding_baseball_bat': 0.5013713836669922, 'white_hair': 0.4999670684337616, 'blush': 0.4860053062438965, 'cloud': 0.474183052778244, 'standing': 0.4724341332912445, 'thigh_gap': 0.4330931305885315, 'short_shorts': 0.39793258905410767, 'parted_lips': 0.36694538593292236, 'crop_top_overhang': 0.3321989178657532, 'official_alternate_costume': 0.3157039284706116, 'blurry': 0.24181532859802246, 'groin': 0.21906554698944092, 'ass_visible_through_thighs': 0.2188207507133484, 'cropped_shirt': 0.18700966238975525, 'taut_shirt': 0.08612403273582458, 'taut_clothes': 0.0701744556427002}
        >>> character
        {'skadi_(arknights)': 0.9796262979507446}
        >>> rating
        {'sensitive': 0.9580697417259216}
        >>>
        >>> # get rating and character only
        >>> rating, character = multilabel_timm_predict(
        ...     'skadi.jpg',
        ...     repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
        ...     fmt=('rating', 'character'),
        ... )
        >>> rating
        {'sensitive': 0.9580697417259216}
        >>> character
        {'skadi_(arknights)': 0.9796262979507446}
        >>>
        >>> # get embeddings only
        >>> embedding = multilabel_timm_predict(
        ...     'skadi.jpg',
        ...     repo_id='animetimm/mobilenetv3_large_150d.dbv4-full',
        ...     fmt='embedding',
        ... )
        >>> embedding.dtype, embedding.shape
        (dtype('float32'), (1280,))
    """
    model = _open_models_for_repo_id(
        repo_id=repo_id,
        hf_token=hf_token,
    )
    return model.predict(
        image=image,
        preprocessor=preprocessor,
        thresholds=thresholds,
        use_tag_thresholds=use_tag_thresholds,
        fmt=fmt,
    )
