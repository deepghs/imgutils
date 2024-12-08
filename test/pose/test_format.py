import pytest

from imgutils.pose import OP18KeyPointSet


@pytest.fixture()
def demo_pose18(pose_data_rin):
    return OP18KeyPointSet(pose_data_rin)


def _get_xy_range(data):
    return data[:, 0].min(), data[:, 0].max(), data[:, 1].min(), data[:, 1].max()


@pytest.mark.unittest
class TestPoseFormat:
    def test_pose18_points(self, demo_pose18):
        assert _get_xy_range(demo_pose18.face) == \
               pytest.approx((465.76692708333326, 582.9609375, 171.5625, 276.2890625))
        assert _get_xy_range(demo_pose18.body) == \
               pytest.approx((321.14453125, 804.8815104166667, 186.5234375, 1368.4375))
        assert _get_xy_range(demo_pose18.left_hand) == \
               pytest.approx((809.8684895833335, 879.6861979166665, 717.63671875, 772.4934895833335))
        assert _get_xy_range(demo_pose18.right_hand) == \
               pytest.approx((253.8203125, 316.1575520833335, 725.1171875, 787.4544270833335))
        assert _get_xy_range(demo_pose18.left_foot) == \
               pytest.approx((697.6614583333335, 742.5442708333335, 1323.5546875, 1420.80078125))
        assert _get_xy_range(demo_pose18.right_foot) == \
               pytest.approx((453.2994791666665, 500.67578125, 1395.8658854166665, 1483.1380208333335))

    def test_pose18_mul(self, demo_pose18):
        pose = demo_pose18 * 1.2
        assert _get_xy_range(pose.face) == \
               pytest.approx((558.9203124999999, 699.553125, 205.875, 331.546875))
        assert _get_xy_range(pose.body) == \
               pytest.approx((385.37343749999997, 965.8578125, 223.828125, 1642.125))
        assert _get_xy_range(pose.left_hand) == \
               pytest.approx((971.8421875000001, 1055.6234374999997, 861.1640625, 926.9921875000001))
        assert _get_xy_range(pose.right_hand) == \
               pytest.approx((304.58437499999997, 379.3890625000002, 870.140625, 944.9453125000001))
        assert _get_xy_range(pose.left_foot) == \
               pytest.approx((837.1937500000001, 891.0531250000001, 1588.265625, 1704.9609375))
        assert _get_xy_range(pose.right_foot) == \
               pytest.approx((543.9593749999998, 600.8109375, 1675.0390624999998, 1779.7656250000002))

        with pytest.raises(TypeError):
            _ = demo_pose18 * 'st'

    def test_pose18_div(self, demo_pose18):
        pose = demo_pose18 / 1.2
        assert _get_xy_range(pose.face) == \
               pytest.approx((388.1391059027777, 485.80078125, 142.96875, 230.24088541666669))
        assert _get_xy_range(pose.body) == \
               pytest.approx((267.62044270833337, 670.734592013889, 155.43619791666669, 1140.3645833333335))
        assert _get_xy_range(pose.left_hand) == \
               pytest.approx((674.8904079861113, 733.0718315972222, 598.0305989583334, 643.7445746527779))
        assert _get_xy_range(pose.right_hand) == \
               pytest.approx((211.51692708333334, 263.46462673611126, 604.2643229166667, 656.2120225694446))
        assert _get_xy_range(pose.left_foot) == \
               pytest.approx((581.3845486111113, 618.7868923611113, 1102.9622395833335, 1184.0006510416667))
        assert _get_xy_range(pose.right_foot) == \
               pytest.approx((377.7495659722221, 417.22981770833337, 1163.2215711805554, 1235.9483506944446))

        with pytest.raises(TypeError):
            _ = demo_pose18 / 'st'
