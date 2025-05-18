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
    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
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
        image = load_image(image, force_background='white', mode='RGB')
        model = self._open_model()

        val_trans, test_trans = self._open_preprocess()
        if preprocessor == 'test':
            trans = test_trans
        elif preprocessor == 'val':
            trans = val_trans
        else:
            raise ValueError(f'Unknown processor - {preprocessor!r}.')

        input_ = trans(image)[None, ...]
        output_names = [output.name for output in model.get_outputs()]
        output_values = model.run(output_names, {'input': input_})
        return {name: value[0] for name, value in zip(output_names, output_values)}

    def predict(self, image: ImageTyping, preprocessor: Literal['test', 'val'] = 'test', fmt='scores-top5'):
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
    return ClassifyTIMMModel(
        repo_id=repo_id,
        hf_token=hf_token,
    )


def classify_timm_predict(image: ImageTyping, repo_id: str, preprocessor: Literal['test', 'val'] = 'test',
                          fmt='scores-top5', hf_token: Optional[str] = None):
    model = _open_models_for_repo_id(
        repo_id=repo_id,
        hf_token=hf_token,
    )
    return model.predict(
        image=image,
        preprocessor=preprocessor,
        fmt=fmt,
    )
