# imgutils

[![PyPI](https://img.shields.io/pypi/v/dghs-imgutils)](https://pypi.org/project/dghs-imgutils/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dghs-imgutils)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/8bfaa96eaa25cc9dac54debbf22d363d/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/8bfaa96eaa25cc9dac54debbf22d363d/raw/comments.json)

[![Code Test](https://github.com/deepghs/imgutils/workflows/Code%20Test/badge.svg)](https://github.com/deepghs/imgutils/actions?query=workflow%3A%22Code+Test%22)
[![Package Release](https://github.com/deepghs/imgutils/workflows/Package%20Release/badge.svg)](https://github.com/deepghs/imgutils/actions?query=workflow%3A%22Package+Release%22)
[![codecov](https://codecov.io/gh/deepghs/imgutils/branch/main/graph/badge.svg?token=XJVDP4EFAT)](https://codecov.io/gh/deepghs/imgutils)

![GitHub Org's stars](https://img.shields.io/github/stars/deepghs)
[![GitHub stars](https://img.shields.io/github/stars/deepghs/imgutils)](https://github.com/deepghs/imgutils/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/deepghs/imgutils)](https://github.com/deepghs/imgutils/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/deepghs/imgutils)
[![GitHub issues](https://img.shields.io/github/issues/deepghs/imgutils)](https://github.com/deepghs/imgutils/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/deepghs/imgutils)](https://github.com/deepghs/imgutils/pulls)
[![Contributors](https://img.shields.io/github/contributors/deepghs/imgutils)](https://github.com/deepghs/imgutils/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/deepghs/imgutils)](https://github.com/deepghs/imgutils/blob/master/LICENSE)

Utilities for Images


## How to check truncated image

```python
from imgutils.validate import is_truncated_file

if __name__ == '__main__':
    filename = 'test_jpg.jpg'
    if is_truncated_file(filename):
        print('This image is truncated, you\'d better '
              'remove this shit from your dataset.')
    else:
        print('This image is okay!')

```