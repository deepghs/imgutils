import json
import os
import tempfile

import onnx
import torch
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory

from imgutils.preprocess import parse_pillow_transforms
from zoo.utils import onnx_optimize
from .demo import load_model, get_dummy_input, load_tags_list, get_preprocessor, TORCH_DTYPE


def export_model_to_onnx(model, onnx_filename, opset_version: int = 17, verbose: bool = True,
                         no_optimize: bool = False):
    dummy_input = torch.from_numpy(get_dummy_input()).to('cpu').type(TORCH_DTYPE)
    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_file,
            verbose=verbose,
            input_names=["input"],
            output_names=["prediction"],

            opset_version=opset_version,
            dynamic_axes={
                "input": {0: "batch"},
                "prediction": {0: "batch"},
            }
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, onnx_filename)


def export(repository: str):
    with TemporaryDirectory() as upload_dir:
        model = load_model(no_tags=True)
        # print(model.tags)
        # del model.tags
        # del model.num_tags
        export_model_to_onnx(
            model=model,
            onnx_filename=os.path.join(upload_dir, 'model.onnx')
        )

        df_tags = load_tags_list()
        df_tags.to_csv(os.path.join(upload_dir, 'tags.csv'), index=False)

        with open(os.path.join(upload_dir, 'preprocessor.json'), 'w') as f:
            json.dump({
                'stages': parse_pillow_transforms(get_preprocessor()),
            }, f, indent=4, sort_keys=True, ensure_ascii=False)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='model',
            local_directory=upload_dir,
            path_in_repo='.',
            message='Syncing deepgelbooru ONNX model',
        )


if __name__ == '__main__':
    export(
        repository='deepghs/deepgelbooru_onnx',
    )
