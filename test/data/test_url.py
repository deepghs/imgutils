import pytest

from imgutils.data import is_http_url, download_image_from_url, load_image
from imgutils.data.url import _is_github_url, _process_github_url_for_downloading, _is_hf_url, \
    _process_hf_url_for_downloading
from test.testings import get_testfile


@pytest.mark.unittest
class TestDataURL:
    @pytest.mark.parametrize(['url', 'local_image'], [
        (
                'https://github.com/deepghs/imgutils/blob/main/test/testfile/nian_640.png',
                ('nian_640.png',)
        ),
        (
                'https://huggingface.co/deepghs/eattach_monochrome_experiments/blob/main/mlp_layer1_seed1/plot_confusion.png',
                ('plot_confusion.png',)
        )
    ])
    def test_download_image_from_url(self, url, local_image, image_diff):
        local_image_file = get_testfile(*local_image)
        actual_image = download_image_from_url(url)
        expected_image = load_image(local_image_file, mode='RGB', force_background='white')
        assert image_diff(
            load_image(actual_image, mode='RGB', force_background='white'),
            expected_image,
            throw_exception=False,
        ) < 1e-2

    def test_is_http_url(self):
        assert is_http_url('http://example.com')
        assert is_http_url('https://example.com')
        assert not is_http_url('ftp://example.com')
        assert not is_http_url('not_a_url')
        assert not is_http_url(123)

    def test_is_github_url(self):
        assert _is_github_url('https://github.com/user/repo')
        assert not _is_github_url('https://gitlab.com/user/repo')

    def test_process_github_url_for_downloading(self):
        url = 'https://github.com/user/repo'
        result = _process_github_url_for_downloading(url)
        assert result == 'https://github.com/user/repo?raw=True'

    def test_is_hf_url(self):
        assert _is_hf_url('https://huggingface.co/user/repo')
        assert _is_hf_url('https://hf.co/user/repo')
        assert not _is_hf_url('https://example.com/user/repo')

    @pytest.mark.parametrize("url, expected", [
        ('https://huggingface.co/datasets/user/repo/blob/main/file.txt',
         'https://huggingface.co/datasets/user/repo/resolve/main/file.txt'),
        ('https://huggingface.co/user/repo/blob/main/file.txt',
         'https://huggingface.co/user/repo/resolve/main/file.txt'),
        ('https://huggingface.co/spaces/user/repo/blob/main/file.txt',
         'https://huggingface.co/spaces/user/repo/resolve/main/file.txt'),
        ('https://huggingface.co/user/repo/resolve/main/file.txt',
         'https://huggingface.co/user/repo/resolve/main/file.txt'),

        ('https://hf.co/datasets/user/repo/blob/main/file.txt',
         'https://hf.co/datasets/user/repo/resolve/main/file.txt'),
        ('https://hf.co/user/repo/blob/main/file.txt',
         'https://hf.co/user/repo/resolve/main/file.txt'),
        ('https://hf.co/spaces/user/repo/blob/main/file.txt',
         'https://hf.co/spaces/user/repo/resolve/main/file.txt'),
        ('https://hf.co/user/repo/resolve/main/file.txt',
         'https://hf.co/user/repo/resolve/main/file.txt'),
    ])
    def test_process_hf_url_for_downloading(self, url, expected):
        assert _process_hf_url_for_downloading(url) == expected

    def test_process_hf_url_for_downloading_invalid(self):
        with pytest.raises(ValueError, match="Unsupported huggingface URL"):
            _process_hf_url_for_downloading('https://huggingface.co/user/repo/invalid/path')
