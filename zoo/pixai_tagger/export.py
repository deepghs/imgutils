import json
import os.path

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client

from imgutils.preprocess import parse_torchvision_transforms
from zoo.pixai_tagger.tags import load_tags
from .min_script import EndpointHandler


def sync(src_repo: str, dst_repo: str):
    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=dst_repo, repo_type='model'):
        hf_client.create_repo(repo_id=dst_repo, repo_type='model', private=True)

    handler = EndpointHandler(repo_id=src_repo)
    with TemporaryDirectory() as upload_dir:
        preprocessor = handler.transform
        preprocessor_file = os.path.join(upload_dir, 'preprocessor.json')
        logging.info(f'Dumping preprocessor:\n{preprocessor}\nto file {preprocessor_file!r}.')
        with open(preprocessor_file, 'w') as f:
            json.dump({
                'stages': parse_torchvision_transforms(handler.transform),
            }, f, sort_keys=True, ensure_ascii=False, indent=4)

        logging.info('Scanning tags ...')
        categories = np.zeros((len(handler.index_to_tag_map),), dtype=np.int32)
        idx = np.array(range(len(categories)))
        categories[idx < handler.gen_tag_count] = 0
        categories[idx >= handler.gen_tag_count] = 4
        df_src_tags = pd.DataFrame({
            'name': [v for _, v in sorted(handler.index_to_tag_map.items())],
            'category': categories,
        })
        df_tags = load_tags(df_src_tags)
        exts = []
        for titem in df_tags.to_dict('records'):
            if titem['category'] == 4:
                if titem['name'] in handler.character_ip_mapping:
                    exts.append(handler.character_ip_mapping[titem['name']])
                else:
                    exts.append([])
            else:
                exts.append([])
        df_tags['ips'] = exts
        logging.info(f'Tags:\n{df_tags}')
        df_tags.to_csv(os.path.join(upload_dir, 'selected_tags.csv'), index=False)

        with open(os.path.join(upload_dir, 'categories.json'), 'w') as f:
            json.dump([
                {
                    "category": 0,
                    "name": "general"
                },
                {
                    "category": 4,
                    "name": "character"
                },
            ], f, sort_keys=True, ensure_ascii=False, indent=4)
        df_th = pd.DataFrame([
            {'category': 0, 'name': 'general', 'threshold': handler.default_general_threshold},
            {'category': 4, 'name': 'character', 'threshold': handler.default_character_threshold},
        ])
        df_th.to_csv(os.path.join(upload_dir, 'thresholds.csv'), index=False)

        handler.model

        os.system(f'tree {upload_dir!r}')
        input()
    # print(df_tags)

    # pprint(handler.index_to_tag_map)
    # pprint(handler.gen_tag_count)
    # pprint(handler.character_tag_count)
    # pprint(handler.character_ip_mapping)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        src_repo='pixai-labs/pixai-tagger-v0.9',
        dst_repo='deepghs/pixai-tagger-v0.9-onnx'
    )
