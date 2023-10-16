import os.path

import matplotlib.font_manager as fm


def get_cn_fp() -> fm.FontProperties:
    ttf_file = os.path.join(os.path.dirname(__file__), 'SimHei.ttf')
    return fm.FontProperties(fname=ttf_file)
