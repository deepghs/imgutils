import numpy as np
import pytest

from test.testings import get_testfile


@pytest.fixture()
def img_file_2girls():
    return get_testfile('pose', '2girls.png')


@pytest.fixture()
def pose_data_2girls_0():
    return np.load(get_testfile('pose', '2girls_pose_0.npy'))


@pytest.fixture()
def pose_data_2girls_1():
    return np.load(get_testfile('pose', '2girls_pose_1.npy'))


@pytest.fixture()
def img_file_halfbody():
    return get_testfile('pose', 'halfbody.png')


@pytest.fixture()
def pose_data_halfbody():
    return np.load(get_testfile('pose', 'halfbody_pose.npy'))


@pytest.fixture()
def img_file_rin():
    return get_testfile('pose', 'tohsaka_rin.png')


@pytest.fixture()
def pose_data_rin():
    return np.load(get_testfile('pose', 'tohsaka_rin_pose.npy'))


@pytest.fixture()
def pose_data_nad_rin():
    return np.load(get_testfile('pose', 'tohsaka_rin_nad_pose.npy'))
