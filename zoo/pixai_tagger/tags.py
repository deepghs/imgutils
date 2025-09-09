from functools import lru_cache

import pandas as pd
from ditk import logging
from hfutils.operate import get_hf_client
from tqdm import tqdm
from waifuc.utils import srequest

from zoo.wd14.tags import _get_tag_by_name, _db_session

_CATEGORY_MAPS = {
    'general': 0,
    'character': 4,
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


def load_tags(df_src_tags: pd.DataFrame):
    hf_client = get_hf_client()
    df_p_tags = pd.read_csv(hf_client.hf_hub_download(
        repo_id='deepghs/site_tags',
        repo_type='dataset',
        filename='danbooru.donmai.us/tags.csv'
    ))
    logging.info(f'Loaded danbooru tags pool, columns: {df_p_tags.columns!r}')
    d_p_tags = {(item['category'], item['name']): item for item in df_p_tags.to_dict('records')}

    rows = []
    src_tags = df_src_tags.to_dict('records')
    for i in tqdm(range(len(src_tags)), desc='Scan Tags'):
        tag_name = src_tags[i]['name']
        category = src_tags[i]['category']
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
                if category != tag_info['category']:
                    logging.warning(f'Category not match for tag {tag_name!r}, '
                                    f'replace category {category!r} --> {tag_info["category"]!r}')
                    category = tag_info['category']
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
    return df


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    df = load_tags()
    df.to_csv('test_df.csv', index=False)
