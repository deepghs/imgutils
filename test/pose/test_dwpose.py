import numpy as np
import pytest

from imgutils.pose import dwpose_estimate, OP18KeyPointSet


@pytest.mark.unittest
class TestPoseDwpose:
    def test_dwpose_estimate(self, img_file_rin, pose_data_rin):
        poses = dwpose_estimate(img_file_rin)
        assert len(poses) == 1
        assert isinstance(poses[0], OP18KeyPointSet)
        assert np.isclose(pose_data_rin, poses[0].all).all()

    def test_dwpose_estimate_no_auto_detect(self, img_file_rin, pose_data_nad_rin):
        poses = dwpose_estimate(img_file_rin, auto_detect=False)
        assert len(poses) == 1
        assert isinstance(poses[0], OP18KeyPointSet)
        assert np.isclose(pose_data_nad_rin, poses[0].all).all()

    def test_dwpose_estimate_halfbody(self, img_file_halfbody, pose_data_halfbody):
        poses = dwpose_estimate(img_file_halfbody)
        assert len(poses) == 1
        assert isinstance(poses[0], OP18KeyPointSet)
        assert np.isclose(pose_data_halfbody, poses[0].all).all()

    def test_dwpose_estimate_2girls(self, img_file_2girls, pose_data_2girls_0, pose_data_2girls_1):
        poses = dwpose_estimate(img_file_2girls)
        assert len(poses) == 2
        assert isinstance(poses[0], OP18KeyPointSet)
        assert isinstance(poses[1], OP18KeyPointSet)
        assert np.isclose(pose_data_2girls_0, poses[0].all).all()
        assert np.isclose(pose_data_2girls_1, poses[1].all).all()

    def test_dwpose_estimate_nothing(self, img_file_gun):
        poses = dwpose_estimate(img_file_gun)
        assert len(poses) == 0
