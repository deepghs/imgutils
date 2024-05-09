import glob
import json
import os.path
from functools import lru_cache
from typing import List, Tuple, Dict, Iterator

import numpy as np
import pytest
from hbutils.testing import disable_output
from huggingface_hub import HfFileSystem, HfApi
from natsort import natsorted
from sklearn.metrics import adjusted_rand_score

from imgutils.metrics import ccip_difference, ccip_default_threshold, ccip_extract_feature, ccip_same, ccip_batch_same, \
    ccip_clustering, ccip_merge, ccip_batch_differences
from test.testings import get_testfile


def _all_images_and_cids() -> Tuple[List[str], List[int]]:
    files = natsorted(glob.glob(get_testfile('dataset', 'images_test_v1', '*', '*.jpg')))[::2][:12]
    cids = [int(os.path.basename(os.path.dirname(file))) for file in files]
    return files, cids


@pytest.fixture()
def images_12() -> List[str]:
    return _all_images_and_cids()[0]


@pytest.fixture()
def images_cids() -> List[int]:
    return _all_images_and_cids()[1]


@pytest.fixture()
def img_1(images_12):
    return images_12[0]


@pytest.fixture()
def img_2(images_12):
    return images_12[1]


@pytest.fixture()
def img_3(images_12):
    return images_12[2]


@pytest.fixture()
def img_4(images_12):
    return images_12[3]


@pytest.fixture()
def img_5(images_12):
    return images_12[4]


@pytest.fixture()
def img_6(images_12):
    return images_12[5]


@pytest.fixture()
def img_7(images_12):
    return images_12[6]


@pytest.fixture()
def img_8(images_12):
    return images_12[7]


@pytest.fixture()
def img_9(images_12):
    return images_12[8]


@pytest.fixture()
def img_10(images_12):
    return images_12[9]


@pytest.fixture()
def img_11(images_12):
    return images_12[10]


@pytest.fixture()
def img_12(images_12):
    return images_12[11]


@pytest.fixture()
def threshold() -> float:
    return ccip_default_threshold()


@pytest.fixture()
def s_threshold(threshold) -> float:
    return threshold + 0.05


MERGE_TAGS = [
    'little_red_riding_hood_(grimm)',
    'maria_cadenzavna_eve',
    'misaka_mikoto',
    'dido_(azur_lane)',
    'hina_(dress)_(blue_archive)',
    'warspite_(kancolle)',
    'kallen_kaslana',
    "kal'tsit_(arknights)",
    'anastasia_(fate)',
    "m16a1_(girls'_frontline)",
]

hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))
hf_client = HfApi(token=os.environ.get('HF_TOKEN'))
SRC_REPO = 'deepghs/character_index'


@lru_cache()
def _get_source_list() -> List[dict]:
    return json.loads(hf_fs.read_text(f'datasets/{SRC_REPO}/characters.json'))


@lru_cache()
def _get_source_dict() -> Dict[str, dict]:
    return {item['tag']: item for item in _get_source_list()}


def list_character_tags() -> Iterator[str]:
    for item in _get_source_list():
        yield item['tag']


def get_detailed_character_info(tag: str) -> dict:
    return _get_source_dict()[tag]


def get_np_feats(tag):
    item = get_detailed_character_info(tag)
    return np.load(hf_client.hf_hub_download(
        repo_id=SRC_REPO,
        repo_type='dataset',
        filename=f'{item["hprefix"]}/{item["short_tag"]}/feat.npy'
    ))


@pytest.mark.unittest
class TestMetricCCIP:
    def test_ccip_difference(self, img_1, img_2, img_3, img_4, img_5, img_6, img_7, s_threshold):
        assert ccip_difference(img_1, img_2) < s_threshold
        assert ccip_difference(img_1, img_3) < s_threshold
        assert ccip_difference(img_2, img_3) < s_threshold

        f1, f2, f3 = map(ccip_extract_feature, (img_1, img_2, img_3))
        assert ccip_difference(f1, f2) < s_threshold
        assert ccip_difference(f1, f3) < s_threshold
        assert ccip_difference(f2, f3) < s_threshold

        assert ccip_difference(img_1, img_4) >= s_threshold
        assert ccip_difference(img_1, img_5) >= s_threshold
        assert ccip_difference(img_1, img_6) >= s_threshold
        assert ccip_difference(img_6, img_7) >= s_threshold

    def test_ccip_same(self, img_1, img_2, img_3, img_4, img_5, img_6, img_7, s_threshold):
        assert ccip_same(img_1, img_2, threshold=s_threshold)
        assert ccip_same(img_1, img_3, threshold=s_threshold)
        assert ccip_same(img_2, img_3, threshold=s_threshold)

        f1, f2, f3 = map(ccip_extract_feature, (img_1, img_2, img_3))
        assert ccip_same(f1, f2, threshold=s_threshold)
        assert ccip_same(f1, f3, threshold=s_threshold)
        assert ccip_same(f2, f3, threshold=s_threshold)

        assert not ccip_same(img_1, img_4, threshold=s_threshold)
        assert not ccip_same(img_1, img_5, threshold=s_threshold)
        assert not ccip_same(img_1, img_6, threshold=s_threshold)
        assert not ccip_same(img_6, img_7, threshold=s_threshold)

    def test_ccip_batch_same(self, images_12, images_cids, s_threshold):
        matrix = ccip_batch_same(images_12, threshold=s_threshold)
        cids = np.array(images_cids)
        cmatrix = cids == cids.reshape(-1, 1)

        assert (matrix != cmatrix).sum() < 5

    def test_ccip_cluster(self, images_12, images_cids):
        with disable_output():
            assert adjusted_rand_score(
                ccip_clustering(images_12, min_samples=2),
                images_cids
            ) >= 0.98

        with disable_output():
            assert adjusted_rand_score(
                ccip_clustering(images_12, min_samples=2, method='dbscan'),
                images_cids,
            ) >= 0.98

        with disable_output():
            assert adjusted_rand_score(
                ccip_clustering(images_12, min_samples=2, method='optics_best'),
                images_cids,
            ) >= 0.98

        with pytest.raises(KeyError):
            _ = ccip_clustering(images_12, min_samples=2, method='what_the_fxxk')

    @pytest.mark.parametrize(['tag'], [
        (tag,) for tag in MERGE_TAGS
    ])
    def test_ccip_merge(self, tag):
        feats = get_np_feats(tag)
        merged_emb = ccip_merge(feats)
        assert ccip_batch_differences([merged_emb, *feats])[0, 1:].mean() <= 0.085
        assert ccip_batch_same([merged_emb, *feats])[0, 1:].all()
