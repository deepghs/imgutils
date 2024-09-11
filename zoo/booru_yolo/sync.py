import json
import os
import shutil

import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.utils import hf_fs_path, parse_hf_fs_path, hf_normpath
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from tqdm import tqdm
from ultralytics import YOLO


def sync(repository: str):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    if hf_fs.exists(hf_fs_path(
            repo_id=repository,
            repo_type='model',
            filename='models.json',
    )):
        models = json.loads(hf_fs.read_text(hf_fs_path(
            repo_id=repository,
            repo_type='model',
            filename='models.json',
        )))
    else:
        models = []

    def _make_readme(readme_file):
        df_models = pd.DataFrame(models)
        with open(readme_file, 'w') as f:
            print(f'---', file=f)
            print(f'pipeline_tag: object-detection', file=f)
            print(f'tags:', file=f)
            print(f'- art', file=f)
            print(f'- anime', file=f)
            print(f'language:', file=f)
            print(f'- en', file=f)
            print(f'library_name: dghs-imgutils', file=f)
            print(f'---', file=f)
            print(f'', file=f)

            print(f'PyTorch and ONNX models of project '
                  f'[aperveyev/booru_yolo](https://github.com/aperveyev/booru_yolo).', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df_models), "model")} in total.', file=f)
            print(f'', file=f)
            print(df_models.to_markdown(), file=f)
            print(f'', file=f)

    for pt_path in tqdm(hf_fs.glob(hf_fs_path(
            repo_id=repository,
            repo_type='model',
            filename='*.pt',
    )), desc='Syncing PT Models'):
        pt_filename = parse_hf_fs_path(pt_path).filename
        logging.info(f'Syncing {pt_filename!r} ...')
        pt_file = hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='model',
            filename=pt_filename,
        )

        model_name, _ = os.path.splitext(pt_filename)
        with TemporaryDirectory() as td:
            model_dir = os.path.join(td, model_name)
            os.makedirs(model_dir, exist_ok=True)

            model_file = os.path.join(model_dir, 'model.pt')
            shutil.copyfile(pt_file, model_file)

            yolo = YOLO(model_file)
            logging.info(f'Names of model: {yolo.names}')
            labels = [None] * (max(yolo.names.keys()) + 1)
            for key, value in yolo.names.items():
                labels[key] = value
            logging.info(f'Labels of model: {labels!r}')
            with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
                json.dump({
                    'name': model_name,
                    'labels': labels,
                }, f, indent=4, sort_keys=True, ensure_ascii=False)

            yolo.export(format='onnx', dynamic=True, simplify=True, opset=14)

            models.append({
                'name': model_name,
                'labels': labels,
            })

            with open(os.path.join(td, 'models.json'), 'w') as f:
                json.dump(models, f, indent=4, sort_keys=True, ensure_ascii=False)
            _make_readme(os.path.join(td, "README.md"))

            operations = []
            for root, _, files in os.walk(td):
                for file in files:
                    full_file = os.path.abspath(os.path.join(root, file))
                    root_file = os.path.abspath(td)
                    operations.append(CommitOperationAdd(
                        path_in_repo=hf_normpath(os.path.relpath(full_file, root_file)),
                        path_or_fileobj=full_file,
                    ))
            operations.append(CommitOperationDelete(
                path_in_repo=pt_filename,
            ))
            hf_client.create_commit(
                repo_id=repository,
                repo_type='model',
                operations=operations,
                commit_message=f'Add model {model_name!r}',
            )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository='deepghs/booru_yolo'
    )
