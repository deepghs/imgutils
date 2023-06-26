import glob
import os.path
from typing import List, Tuple

import numpy as np
import pytest
from hbutils.testing import disable_output
from natsort import natsorted
from sklearn.metrics import adjusted_rand_score

from imgutils.metrics import ccip_difference, ccip_default_threshold, ccip_extract_feature, ccip_same, ccip_batch_same, \
    ccip_clustering
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
