import pytest

from imgutils.data import load_image
from imgutils.detect import detection_visualize
from test.testings import get_testfile


@pytest.mark.unittest
class TestDetectVisual:
    def test_detection_visualize(self, image_diff):
        image = get_testfile('genshin_post.jpg')
        visual = detection_visualize(image, [
            ((202, 155, 356, 294), 'first', 0.878),
            ((938, 87, 1121, 262), 'second', 0.846),
            ((652, 440, 725, 514), 'third', 0.839),
            ((464, 250, 535, 326), 'fourth', 0.765)
        ], fontsize=24)

        assert image_diff(
            load_image(get_testfile('genshin_post_face_visual.jpg'), mode='RGB'),
            visual.convert('RGB'),
            throw_exception=False
        ) < 1e-2
