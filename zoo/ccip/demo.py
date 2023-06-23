import argparse
import os

import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
from tqdm.auto import tqdm

from imgutils.data import load_image
from .dataset import TEST_TRANSFORM
from .model import CCIP


def _load_remote_ckpt(remote_ckpt):
    return hf_hub_download('deepghs/ccip', remote_ckpt, repo_type='model')


def _get_model_from_ckpt(model_name, ckpt, device, fp16: bool):
    model = CCIP(model_name).to(device)
    model.eval()
    if fp16:
        model = model.half()

    state = torch.load(ckpt, map_location='cpu')
    try:
        model.load_state_dict(state)
    except:
        len_p = len('module._orig_mod.')
        model.load_state_dict({k[len_p:]: v for k, v in state.items()})

    preprocess = transforms.Compose(TEST_TRANSFORM + model.preprocess)

    return model, preprocess


class Infer:
    def __init__(self, args, device=None):
        self.args = args
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.img_transform = _get_model_from_ckpt(
            model_name=args.model_name,
            ckpt=args.ckpt or _load_remote_ckpt(args.remote_ckpt),
            device=self.device,
            fp16=args.fp16
        )

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

    @torch.no_grad()
    def infer_batch(self, img_list, bs=8):
        feat_list = []
        for i in tqdm(range(0, len(img_list), bs)):
            imgs = torch.stack(img_list[i:i+bs]).to(self.device)
            if self.args.fp16:
                imgs = imgs.half()
            feat = self.model.feature(imgs)
            feat_list.append(feat)
        feat_list = torch.cat(feat_list, dim=0)
        outputs = self.model.metrics(feat_list)
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
    torch.set_printoptions(precision=2,sci_mode=False)
    imgs = []
    # imgs.append(demo.load_img(
    #     r'E:\dataset\pixiv\ganyu/11ee873afc5aacff2fd96248c1820c9240e922f6.jpg@942w_1320h_progressive.webp'))
    # imgs.append(demo.load_img(
    #     r'E:\dataset\pixiv\ganyu/91039559171fd81f1ccb54838e1f546a4c3d6e7c.jpg@942w_942h_progressive.webp'))
    # imgs.append(
    #     demo.load_img(r'E:\dataset\pixiv\p1/eb7009f1dd5ecc61cf8d55f7d82c1922487b3cfc.jpg@942w_1338h_progressive.webp'))
    # imgs.append(demo.load_img(r'E:\dataset\pixiv\p1/c398774304db7cc737bb57fa2f380295.jpg'))
    # imgs.append(demo.load_img(r'E:\dataset\pixiv\p1/20221215165339_13707.png'))
    # imgs.append(demo.load_img(r'E:\dataset\pixiv\p1/e768ebbb4a116c8b85b39342d3348775.png'))
    root=r'E:\dataset\ccip_error_images'
    for path in os.listdir(root):
        print(path)
        imgs.append(demo.load_img(os.path.join(root,path)))
    pred = demo.infer_one(imgs)
    print(pred)
