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
               pytest.approx((463.900390625, 578.451171875, 169.853515625, 276.93359375))
        assert _get_xy_range(demo_pose18.body) == \
               pytest.approx((321.95703125, 802.572265625, 184.794921875, 1365.166015625))
        assert _get_xy_range(demo_pose18.left_hand) == \
               pytest.approx((807.552734375, 872.298828125, 715.21484375, 777.470703125))
        assert _get_xy_range(demo_pose18.right_hand) == \
               pytest.approx((252.23046875, 316.9765625, 725.17578125, 787.431640625))
        assert _get_xy_range(demo_pose18.left_foot) == \
               pytest.approx((697.982421875, 742.806640625, 1320.341796875, 1419.951171875))
        assert _get_xy_range(demo_pose18.right_foot) == \
               pytest.approx((453.939453125, 501.25390625, 1390.068359375, 1482.20703125))

    def test_pose18_mul(self, demo_pose18):
        pose = demo_pose18 * 1.2
        assert _get_xy_range(pose.face) == \
               pytest.approx((556.6804687499999, 694.1414062499999, 203.82421875, 332.3203125))
        assert _get_xy_range(pose.body) == \
               pytest.approx((386.3484375, 963.0867187499999, 221.75390625, 1638.19921875))
        assert _get_xy_range(pose.left_hand) == \
               pytest.approx((969.0632812499999, 1046.75859375, 858.2578125, 932.96484375))
        assert _get_xy_range(pose.right_hand) == \
               pytest.approx((302.6765625, 380.371875, 870.2109375, 944.91796875))
        assert _get_xy_range(pose.left_foot) == \
               pytest.approx((837.5789062499999, 891.3679687499999, 1584.41015625, 1703.94140625))
        assert _get_xy_range(pose.right_foot) == \
               pytest.approx((544.7273437499999, 601.5046874999999, 1668.08203125, 1778.6484375))

        with pytest.raises(TypeError):
            _ = demo_pose18 * 'st'

    def test_pose18_div(self, demo_pose18):
        pose = demo_pose18 / 1.2
        assert _get_xy_range(pose.face) == \
               pytest.approx((386.5836588541667, 482.0426432291667, 141.54459635416669, 230.77799479166669))
        assert _get_xy_range(pose.body) == \
               pytest.approx((268.2975260416667, 668.8102213541667, 153.99576822916669, 1137.6383463541667))
        assert _get_xy_range(pose.left_hand) == \
               pytest.approx((672.9606119791667, 726.9156901041667, 596.0123697916667, 647.8922526041667))
        assert _get_xy_range(pose.right_hand) == \
               pytest.approx((210.19205729166669, 264.1471354166667, 604.3131510416667, 656.1930338541667))
        assert _get_xy_range(pose.left_foot) == \
               pytest.approx((581.6520182291667, 619.0055338541667, 1100.2848307291667, 1183.2926432291667))
        assert _get_xy_range(pose.right_foot) == \
               pytest.approx((378.2828776041667, 417.7115885416667, 1158.3902994791667, 1235.1725260416667))

        with pytest.raises(TypeError):
            _ = demo_pose18 / 'st'
