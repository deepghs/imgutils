import os.path
import re
from typing import Optional
from urllib.parse import urljoin

import httpx
import numpy as np
import pandas as pd
from ditk import logging
from hbutils.system import urlsplit, TemporaryDirectory
from hfutils.operate import get_hf_client, upload_directory_as_directory
from pyquery import PyQuery as pq
from tqdm import tqdm
from waifuc.utils import srequest

from zoo.wd14.tags import _db_session, _get_tag_by_name

logging.try_init_root(logging.INFO)
session = _db_session()


def _get_tag_name_from_wiki_url(wiki_url):
    try:
        resp = srequest(session, 'GET', wiki_url)
    except httpx.HTTPStatusError as err:
        if err.response.status_code == 404:
            return None
        raise

    tag_url = urljoin(str(resp.url), pq(resp.text)('h1#wiki-page-title a').attr('href'))
    segments = list(filter(bool, urlsplit(tag_url).path_segments))
    if segments == ['posts', ]:
        return urlsplit(tag_url).query_dict['tags']
    else:
        return None


def _get_tags_from_wiki(wiki_name):
    resp = srequest(session, 'GET', f'https://danbooru.donmai.us/wiki_pages/{wiki_name}')
    page = pq(resp.text)
    body = page('#wiki-page-body')
    body.remove('details')

    retval = []
    if _get_tag_by_name(wiki_name):
        retval.append(wiki_name)

    for item in tqdm(list(body('li > a').items()), desc=f'Wiki {wiki_name!r}'):
        url = urljoin(str(resp.url), item.attr('href'))
        segments = list(filter(bool, urlsplit(url).path_segments))
        if segments[0] == 'wiki_pages':
            tag_name = _get_tag_name_from_wiki_url(url) or segments[-1]
            if tag_name and _get_tag_by_name(tag_name):
                retval.append(tag_name)
    return retval


def _extract_tags_from_ul(ul_element, base_url, last_group_name=None):
    last_one = None
    for ch in ul_element.children().items():
        if str(ch).lstrip().startswith('<li'):
            wiki_url = urljoin(base_url, ch('a').attr('href'))
            segments = list(filter(bool, urlsplit(wiki_url).path_segments))
            assert segments[0] == 'wiki_pages', f'{wiki_url!r}'
            yield segments[1], last_group_name
            last_one = segments[1]

        elif str(ch).lstrip().startswith('<ul'):
            yield from _extract_tags_from_ul(ch, base_url, last_one)

        else:
            raise ValueError(f'Unknown element: {ch!r}')


def _get_other_lsts():
    resp = srequest(
        session, 'GET', 'https://danbooru.donmai.us/wiki_pages.json',
        params={
            'limit': '1000',
            'search[title_normalize]': 'list_of_*'
        }
    )
    return [item['title'] for item in resp.json()]


def _get_other_tag_groups():
    resp = srequest(
        session, 'GET', 'https://danbooru.donmai.us/wiki_pages.json',
        params={
            'limit': '1000',
            'search[title_normalize]': 'tag_group:*'
        }
    )
    return [item['title'] for item in resp.json()]


def _get_groups():
    resp = srequest(session, 'GET', f'https://danbooru.donmai.us/wiki_pages/tag_groups')
    page = pq(resp.text)
    body = page('#wiki-page-body')
    body.remove('details')

    current_title = None
    exist_wikis = set()
    for ch in body.children().items():
        if ch.attr('id'):
            matching = re.fullmatch(r'dtext-(?P<name>[a-zA-Z0-9-_]+)', ch.attr('id'))
            current_title = matching.group('name')
        elif str(ch).lstrip().startswith('<ul'):
            lst = []
            for wiki_name, parent_wiki_name in _extract_tags_from_ul(ch, str(resp.url), None):
                lst.append((wiki_name, parent_wiki_name))
                exist_wikis.add(wiki_name)
            yield current_title, lst

    yield 'other-groups', [(name, None) for name in _get_other_tag_groups() if name not in exist_wikis]
    yield 'other-lists', [(name, None) for name in _get_other_lsts() if name not in exist_wikis]


def _make_table(limit: Optional[int] = None):
    all_groups = {}
    all_tags = {}
    cnt = 0
    for group_category, groups in tqdm(_get_groups(), desc='Groups'):
        for group_name, parent_group_name in groups:
            if group_name not in all_groups:
                all_groups[group_name] = (group_category, parent_group_name or group_category)

            for tag in tqdm(_get_tags_from_wiki(group_name), desc=f'Tags in {group_name!r}'):
                tag_item = _get_tag_by_name(tag)
                tag_name, tag_id, tag_posts, tag_category = (
                    tag_item['name'], tag_item['id'], tag_item['post_count'], tag_item['category'])
                if tag_name not in all_tags:
                    all_tags[tag_name] = (tag_id, tag_name, tag_posts, tag_category, [])
                _, _, _, _, exist_groups = all_tags[tag_name]

                g = group_name
                while g:
                    exist_groups.append(g)
                    if g in all_groups:
                        _, g = all_groups[g]
                    else:
                        break
                exist_groups = sorted(set(exist_groups))
                all_tags[tag_name] = (tag_id, tag_name, tag_posts, tag_category, exist_groups)

                cnt += 1
                if limit and cnt >= limit:
                    break

            if limit and cnt >= limit:
                break

        if limit and cnt >= limit:
            break

    records = []
    for tag_name, (tag_id, _, tag_posts, tag_category, exist_groups) in all_tags.items():
        records.append({
            'id': tag_id,
            'tag': tag_name,
            'posts': tag_posts,
            'category': tag_category,
            **{f'is_{name}': True for name in exist_groups},
        })

    df_record = pd.DataFrame(records)
    df_record = df_record.replace(np.NaN, False)
    df_record = df_record.sort_values(by=['posts', 'id'], ascending=[False, True])

    groupx = []
    for group_name, (group_category, group_parent) in all_groups.items():
        groupx.append({
            'name': group_name,
            'category': group_category,
            'parent': None if group_category == group_parent else group_parent,
        })
    df_groups = pd.DataFrame(groupx)

    return df_record, df_groups


def sync(repository='deepghs/danbooru_tag_groups'):
    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)

    df_record, df_groups = _make_table()
    with TemporaryDirectory() as td:
        df_record.to_csv(os.path.join(td, 'tags.csv'), index=False)
        df_groups.to_csv(os.path.join(td, 'groups.csv'), index=False)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            message='Sync Danbooru Tags'
        )


if __name__ == '__main__':
    sync()
