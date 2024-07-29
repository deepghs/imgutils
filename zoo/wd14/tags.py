import json
import logging
import os
from functools import lru_cache
from typing import List, Set, Dict

import numpy as np
import pandas as pd
from ditk import logging
from hfutils.operate import get_hf_fs
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from waifuc.source import DanbooruSource
from waifuc.utils import srequest

from imgutils.tagging import is_basic_character_tag
from imgutils.tagging.wd14 import MODEL_NAMES, LABEL_FILENAME


@lru_cache()
def _db_session():
    s = DanbooruSource(['1girl'])
    s._prune_session()
    return s.session


@lru_cache(maxsize=65536)
def _get_tag_by_id(tag_id: int):
    session = _db_session()
    return srequest(session, "GET", f'https://danbooru.donmai.us/tags/{tag_id}.json').json()


@lru_cache(maxsize=125536)
def _get_tag_by_name(tag_name: str):
    session = _db_session()
    vs = srequest(
        session, 'GET', f'https://danbooru.donmai.us/tags.json',
        params={'search[name]': tag_name}
    ).json()
    return vs[0] if vs else None


@lru_cache(maxsize=65536)
def _simple_search_related_tags(tag: str) -> List[str]:
    session = _db_session()
    tags = []
    for item in srequest(
            session, 'GET', 'https://danbooru.donmai.us/tag_aliases.json',
            params={
                'search[name_matches]': tag,
            }
    ).json():
        if item['consequent_name'] == tag:
            tags.append(item['antecedent_name'])

    return tags


@lru_cache(maxsize=65536)
def _search_related_tags(tag: str, model_name: str = 'ConvNext') -> List[str]:
    existing_names = _tags_name_set(model_name)
    tags = [tag]
    i = 0
    while i < len(tags):
        append_tags = _simple_search_related_tags(tags[i])
        for tag_ in append_tags:
            if tag_ not in tags and tag_ not in existing_names:
                tags.append(tag_)

        i += 1

    return tags


@lru_cache()
def _tags_list(model_name) -> pd.DataFrame:
    return pd.read_csv(hf_hub_download(MODEL_NAMES[model_name], LABEL_FILENAME))


@lru_cache()
def _tags_name_set(model_name) -> Set[str]:
    return set(_tags_list(model_name)['name'])


@lru_cache()
def _load_all_previous(repository: str = 'deepghs/wd14_tagger_with_embeddings') -> Dict[int, dict]:
    hf_fs = get_hf_fs()
    d = {}
    for path in hf_fs.glob(f'{repository}/**/tags_info.csv'):
        relpath = os.path.relpath(path, f'{repository}')
        df = pd.read_csv(hf_hub_download(
            repo_id=repository,
            repo_type='model',
            filename=relpath,
        )).replace(np.nan, None)
        for item in df.to_dict('records'):
            if item['tag_id'] not in d:
                d[item['tag_id']] = item
    return d


def _make_tag_info(model_name='ConvNext', lazy_mode: bool = False) -> pd.DataFrame:
    with open(hf_hub_download(
            repo_id='deepghs/tags_meta',
            repo_type='dataset',
            filename='attire_tags.json',
    ), 'r') as f:
        attire_tags = json.load(f)

    df = _tags_list(model_name)
    d = _load_all_previous()
    records = []
    for item in tqdm(df.to_dict('records')):
        if lazy_mode and item['tag_id'] in d:
            item = d[item['tag_id']]
        else:
            if item['category'] != 9:
                tag_info = _get_tag_by_id(item['tag_id'])
                item['count'] = tag_info['post_count']
                aliases = _search_related_tags(item['name'], model_name)
                logging.info(f'Aliases {aliases!r} --> {item["name"]!r}')
                item['aliases'] = json.dumps(aliases)
            else:
                item['aliases'] = json.dumps([item['name']])
            item['is_core'] = (item['category'] == 0) and (is_basic_character_tag(item['name']))
            item['is_attire'] = (item['category'] == 0) and (item['name'] in attire_tags)
        records.append(item)

    df_records = pd.DataFrame(records)
    return df_records


if __name__ == "__main__":
    logging.try_init_root(logging.INFO)
    df = _make_tag_info()
    df.to_csv('test_tags_info.csv', index=False)
