import argparse

import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms

from imgutils.data import load_image
from .dataset import TEST_TRANSFORM
from .model import CCIP


def _load_remote_ckpt(remote_ckpt):
    return hf_hub_download('deepghs/ccip', remote_ckpt, repo_type='model')


class Infer:
    def __init__(self, args, device=None):
        self.args = args
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = CCIP(args.model_name).to(device)
        self.model.eval()
        if self.args.fp16:
            self.model = self.model.half()

        state = torch.load(args.ckpt or _load_remote_ckpt(args.remote_ckpt), map_location='cpu')
        try:
            self.model.load_state_dict(state)
        except:
            len_p = len('module._orig_mod.')
            self.model.load_state_dict({k[len_p:]: v for k, v in state.items()})

        self.img_transform = transforms.Compose(TEST_TRANSFORM + self.model.preprocess)

    def load_img(self, path):
        image = load_image(path, mode='RGB')
        image = self.img_transform(image)
        return image

    @torch.no_grad()
    def infer_one(self, img_list):
        imgs = torch.stack(img_list).to(self.device)
        if self.args.fp16:
            imgs = imgs.half()
        outputs = self.model(imgs)
        return outputs

    @staticmethod
    def build_args():
        parser = argparse.ArgumentParser(description='Stable Diffusion Training')
        parser.add_argument('--model_name', type=str, default='caformer')
        parser.add_argument('--ckpt', type=str, default='')
        parser.add_argument('--remote_ckpt', type=str, default='ccip-caformer-2_fp32.ckpt')
        parser.add_argument('--fp16', default=None, action="store_true")
        return parser.parse_args()


if __name__ == '__main__':
    demo = Infer(Infer.build_args())
    imgs = []
    imgs.append(demo.load_img(
        r'E:\dataset\pixiv\ganyu/11ee873afc5aacff2fd96248c1820c9240e922f6.jpg@942w_1320h_progressive.webp'))
    imgs.append(demo.load_img(
        r'E:\dataset\pixiv\ganyu/91039559171fd81f1ccb54838e1f546a4c3d6e7c.jpg@942w_942h_progressive.webp'))
    imgs.append(
        demo.load_img(r'E:\dataset\pixiv\p1/eb7009f1dd5ecc61cf8d55f7d82c1922487b3cfc.jpg@942w_1338h_progressive.webp'))
    imgs.append(demo.load_img(r'E:\dataset\pixiv\p1/c398774304db7cc737bb57fa2f380295.jpg'))
    imgs.append(demo.load_img(r'E:\dataset\pixiv\p1/20221215165339_13707.png'))
    imgs.append(demo.load_img(r'E:\dataset\pixiv\p1/e768ebbb4a116c8b85b39342d3348775.png'))
    pred = demo.infer_one(imgs)
    print(pred)
