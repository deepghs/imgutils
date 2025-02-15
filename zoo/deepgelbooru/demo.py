import os.path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from imgutils.data import load_image
from imgutils.preprocess.pillow import PillowConvertRGB, PillowResize, PillowToTensor, PillowCompose
from test.testings import get_testfile
from .model import DeepDanbooruModel, ModelWithoutTags

TORCH_DTYPE = torch.float32


def load_model(no_tags: bool = False):
    cls = DeepDanbooruModel if not no_tags else ModelWithoutTags
    return cls.from_single_file(
        hf_hub_download(
            repo_id='v2ray/deepgelbooru',
            repo_type='model',
            filename='model_epoch_13.bin',
        ),
        "cpu",
        torch.float32
    )


_PIC_FILE = get_testfile('nude_girl.png')


# _PIC_FILE = get_testfile('6125785.jpg')


def get_preprocessor():
    return PillowCompose([
        PillowConvertRGB(),
        PillowResize((512, 512)),
        PillowToTensor(),
    ])


def get_dummy_input():
    pic = load_image(_PIC_FILE, mode='RGB')
    compose = get_preprocessor()
    return compose(pic).transpose((1, 2, 0))[None, ...].astype(np.float32)


def load_tags_list():
    d_tags = {}
    with open(os.path.join(os.path.dirname(__file__), 'model_tags.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ord_, tag_text = line.split(' ', maxsplit=1)
            ord_ = int(ord_)
            if tag_text.startswith('rating:'):
                category = 9
            elif ord_ >= 6891 and tag_text != 'minecraft':
                category = 4
            else:
                category = 0
            d_tags[ord_] = {
                'tag_id': ord_,
                'name': tag_text,
                'category': category,
            }

    df = pd.DataFrame(list(d_tags.values()))
    df = df.sort_values(by=['tag_id'], ascending=True)
    return df


if __name__ == '__main__':
    model = load_model(no_tags=True)
    print(model)
    # quit()

    x = get_dummy_input()
    torch_x = torch.from_numpy(x).type(TORCH_DTYPE).to('cpu')
    with torch.no_grad():
        r = model(torch_x)
    y = r[0]

    df_tags = load_tags_list()
    print(df_tags)
    d_tags = {item['tag_id']: item for item in df_tags.to_dict('records')}
    # assert len(df_tags) == len(model.tags)
    # for item in df_tags.to_dict('records'):
    #     assert model.tags[item['tag_id']] == item['name'], \
    #         f'Tag #{item["tag_id"]!r} not match, {item["name"]!r} expected but {model.tags[item["tag_id"]]!r} found.'

    for i, prob in sorted(((i, float(prob)) for i, prob in enumerate(y)), key=lambda x: x[1]):
        if prob >= 0.2:
            print(d_tags[i]['name'], "-", prob)
