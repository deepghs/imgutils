import pytest

from imgutils.data import load_image
from imgutils.pose import op18_visualize, OP18KeyPointSet
from test.testings import get_testfile


@pytest.fixture()
def poses_rin(pose_data_rin):
    return [
        OP18KeyPointSet(pose_data_rin),
    ]


@pytest.fixture()
def poses_halfbody(pose_data_halfbody):
    return [
        OP18KeyPointSet(pose_data_halfbody),
    ]


@pytest.fixture()
def poses_2girls(pose_data_2girls_0, pose_data_2girls_1):
    return [
        OP18KeyPointSet(pose_data_2girls_0),
        OP18KeyPointSet(pose_data_2girls_1),
    ]


@pytest.fixture()
def visual_img_rin():
    return load_image(get_testfile('pose', 'tohsaka_rin_visual.png'))


@pytest.fixture()
def visual_img_nad_rin():
    return load_image(get_testfile('pose', 'tohsaka_rin_nad_visual.png'))


@pytest.fixture()
def visual_img_halfbody():
    return load_image(get_testfile('pose', 'halfbody_visual.png'))


@pytest.fixture()
def visual_img_2girls():
    return load_image(get_testfile('pose', '2girls_visual.png'))


@pytest.mark.unittest
class TestPoseVisual:
    def test_op18_visualize_rin(self, img_file_rin, poses_rin, visual_img_rin, image_diff):
        visual_img = op18_visualize(img_file_rin, poses_rin)
        assert min(visual_img.width, visual_img.height) <= 512
        assert image_diff(visual_img, visual_img_rin, throw_exception=False) < 1e-2

    def test_op18_visualize_rin_nad(self, img_file_rin, poses_rin, visual_img_nad_rin, image_diff):
        visual_img = op18_visualize(img_file_rin, poses_rin, min_edge_size=None)
        assert visual_img.size == load_image(img_file_rin).size
        assert image_diff(visual_img, visual_img_nad_rin, throw_exception=False) < 1e-2

    def test_op18_visualize_halfbody(self, img_file_halfbody, poses_halfbody, visual_img_halfbody, image_diff):
        visual_img = op18_visualize(img_file_halfbody, poses_halfbody)
        assert min(visual_img.width, visual_img.height) <= 512
        assert image_diff(visual_img, visual_img_halfbody, throw_exception=False) < 1e-2

    def test_op18_visualize_2girls(self, img_file_2girls, poses_2girls, visual_img_2girls, image_diff):
        visual_img = op18_visualize(img_file_2girls, poses_2girls)
        assert min(visual_img.width, visual_img.height) <= 512
        assert image_diff(visual_img, visual_img_2girls, throw_exception=False) < 1e-2
