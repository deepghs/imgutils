import pytest

from imgutils.data import load_image
from imgutils.restore import remove_adversarial_noise
from test.testings import get_testfile


@pytest.fixture()
def adversarial_input():
    return get_testfile('adversarial', 'adversarial_input.png')


@pytest.fixture()
def adversarial_output_pil():
    return load_image(get_testfile('adversarial', 'adversarial_output.png'))


@pytest.mark.unittest
class TestRestoreAdversarial:
    def test_remove_adversarial_noises(self, adversarial_input, adversarial_output_pil, image_diff):
        assert image_diff(
            remove_adversarial_noise(adversarial_input),
            adversarial_output_pil,
            throw_exception=False
        ) < 5e-3
