import json
from functools import lru_cache

import pandas as pd
from ditk import logging
from hfutils.operate import get_hf_client
from tqdm import tqdm
from waifuc.utils import srequest

from zoo.wd14.tags import _get_tag_by_name, _db_session

meta_repo_id = 'AngelBottomless/camie-tagger-onnxruntime'

_CATEGORY_MAPS = {
    'rating': 9,
    'year': 10,

    'general': 0,
    'artist': 1,
    'copyright': 3,
    'character': 4,
    'meta': 5,
}

_RATING_TAG_IDS = {
    'general': 9999999,
    'sensitive': 9999998,
    'questionable': 9999997,
    'explicit': 9999996,
}


@lru_cache()
def _get_rating_count_by_name(tag_name: str):
    session = _db_session()
    logging.info(f'Getting count for {tag_name!r} ...')
    vs = srequest(
        session, 'GET', f'https://danbooru.donmai.us/counts/posts.json',
        params={'tags': f'rating:{tag_name}'}
    ).json()
    logging.info(f'Result of {tag_name!r}: {vs!r}')
    return vs['counts']['posts']


@lru_cache()
def _get_year_count_by_name(year):
    session = _db_session()
    logging.info(f'Getting count for year {year!r} ...')
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    vs = srequest(
        session, 'GET', f'https://danbooru.donmai.us/counts/posts.json',
        params={'tags': f'date:{start_date}..{end_date}'}
    ).json()
    logging.info(f'Result of year {year!r}: {vs!r}')
    return vs['counts']['posts']


def load_tags():
    hf_client = get_hf_client()
    df_p_tags = pd.read_csv(hf_client.hf_hub_download(
        repo_id='deepghs/site_tags',
        repo_type='dataset',
        filename='danbooru.donmai.us/tags.csv'
    ))
    logging.info(f'Loaded danbooru tags pool, columns: {df_p_tags.columns!r}')
    d_p_tags = {(item['category'], item['name']): item for item in df_p_tags.to_dict('records')}

    with open(hf_client.hf_hub_download(
            repo_id=meta_repo_id,
            repo_type='model',
            filename='metadata.json',
    )) as f:
        d = json.load(f)
        rows = []
        for i in tqdm(range(len(d['idx_to_tag'])), desc='Scan Tags'):
            tag_name = d['idx_to_tag'][str(i)]
            category_str = d['tag_to_category'][tag_name]
            category = _CATEGORY_MAPS[category_str]
            if (category, tag_name) in d_p_tags:
                tag_id = d_p_tags[(category, tag_name)]['id']
                count = d_p_tags[(category, tag_name)]['post_count']
            elif category < 9:
                logging.warning(f'Cannot find tag {tag_name!r}, category: {category!r}.')
                tag_info = _get_tag_by_name(tag_name)
                if tag_info['name'] != tag_name:
                    logging.warning(f'Not found matching tags for {tag_name!r}, will be ignored.')
                    tag_id = -1
                    count = -1
                else:
                    logging.info(f'Tag info found from danbooru - {tag_info!r}.')
                    tag_id = tag_info['id']
                    count = tag_info['post_count']
            elif category == 9:
                tag_name = tag_name.split('_', maxsplit=1)[-1]
                tag_id = _RATING_TAG_IDS[tag_name]
                count = _get_rating_count_by_name(tag_name)
            elif category == 10:
                year_id = int(tag_name.split('_', maxsplit=1)[-1])
                tag_id = 8999999 + 1900 - year_id
                count = _get_year_count_by_name(year_id)
            else:
                logging.warning(f'Unknown tag {tag_name!r} ...')
                tag_id = -1
                count = -1

            rows.append({
                'id': i,
                'tag_id': tag_id,
                'name': tag_name,
                'category': category,
                'count': count,
            })

        df = pd.DataFrame(rows)
        logging.info(f'Tags:\n{df}')
        return df


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    load_tags()
