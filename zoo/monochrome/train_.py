import os.path
import re
from functools import partial
from typing import Optional, Type

import torch
from accelerate import Accelerator
from ditk import logging
from hbutils.random import global_seed
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .alexnet import MonochromeAlexNet
from .dataset import MonochromeDataset, Monochrome2DDataset, random_split_dataset, TRANSFORM_VAL, TRANSFORM2_VAL
from .levit1d import LeSigTransformer
from .levit2d import LeViT
from .loss import FocalLoss
from .resnet1d import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnet2d import ResNet182D, ResNet342D, ResNet502D, ResNet1012D, ResNet1522D
from .transformer import SigTransformer
from .metaformer import CAFormerBuilder
from ..base import _TRAIN_DIR as _GLOBAL_TRAIN_DIR

_TRAIN_DIR = os.path.join(_GLOBAL_TRAIN_DIR, 'monochrome')
_LOG_DIR = os.path.join(_TRAIN_DIR, 'logs')
_CKPT_DIR = os.path.join(_TRAIN_DIR, 'ckpts')

_CKPT_PATTERN = re.compile(r'^monochrome-(?P<name>[a-zA-Z\d_\-]+)-(?P<epoch>\d+)\.ckpt$')

_KNOWN_MODELS = {}
_KNOWN_DIMS = {}
_KNOWN_DATASETS = {}


def _register_model(cls: Type[nn.Module], *args, name=None, **kwargs):
    name = name or cls.__model_name__
    _KNOWN_MODELS[name] = partial(cls, *args, **kwargs)
    _KNOWN_DIMS[name] = getattr(cls, '__dims__', 1)


_register_model(MonochromeAlexNet)
_register_model(ResNet18)
_register_model(ResNet34)
_register_model(ResNet50)
_register_model(ResNet101)
_register_model(ResNet152)
_register_model(ResNet182D)
_register_model(ResNet342D)
_register_model(ResNet502D)
_register_model(ResNet1012D)
_register_model(ResNet1522D)
_register_model(SigTransformer)
_register_model(LeSigTransformer)
_register_model(LeViT)
_register_model(CAFormerBuilder)


def _register_dataset(cls: Type[Dataset]):
    _KNOWN_DATASETS[cls.__dims__] = cls


_register_dataset(MonochromeDataset)
_register_dataset(Monochrome2DDataset)


def _find_latest_ckpt(name: str) -> Optional[str]:
    if os.path.exists(_CKPT_DIR):
        ckpts = []
        for filename in os.listdir(_CKPT_DIR):
            matching = _CKPT_PATTERN.fullmatch(os.path.basename(filename))
            if matching and matching.group('name') == name:
                ckpts.append((int(matching.group('epoch')), os.path.join(_CKPT_DIR, filename)))

        ckpts = sorted(ckpts)
        if ckpts:
            return ckpts[-1][1]
        else:
            return None
    else:
        return None


def _ckpt_epoch(filename: Optional[str]) -> Optional[int]:
    if filename is not None:
        matching = _CKPT_PATTERN.fullmatch(os.path.basename(filename))
        if matching:
            return int(matching.group('epoch'))
        else:
            return None
    else:
        return None


def train(dataset_dir: str, session_name: Optional[str] = None, from_ckpt: Optional[str] = None,
          train_ratio: float = 0.8, batch_size: int = 4, feature_bins: int = 180, fc: Optional[int] = 75,
          max_epochs: int = 500, learning_rate: float = 0.001, weight_decay: float = 1e-3, preference: float = 0.0,
          num_workers: Optional[int] = 8, save_per_epoch: int = 10, eval_epoch: int = 5,
          model_name: str = 'alexnet', seed: Optional[int] = 0):
    if seed is not None:
        # native random, numpy, torch and faker's seeds are includes
        # if you need to register more library for seeding, see:
        # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
        global_seed(seed)

    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    session_name = session_name or model_name
    model_dims = _KNOWN_DIMS[model_name]
    _log_dir = os.path.join(_LOG_DIR, session_name)

    if accelerator.is_local_main_process:
        os.makedirs(_log_dir, exist_ok=True)
        os.makedirs(_CKPT_DIR, exist_ok=True)
        writer = SummaryWriter(_log_dir)
        writer.add_custom_scalars({
            "general": {
                "accuracy": ["Multiline", ["train/accuracy", "test/accuracy"]],
                "false": ["Multiline", ["test/fn", "test/fp", "train/fn", "train/fp"]],
            },
            "test": {
                "false": ["Multiline", ["test/fn", "test/fp"]],
            },
            "train": {
                "false": ["Multiline", ["train/fn", "train/fp"]],
            },
        })
    else:
        writer = None

    # Initialize dataset
    full_dataset = _KNOWN_DATASETS[model_dims](dataset_dir, bins=feature_bins, fc=fc)
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # 使用 random_split 函数拆分数据集
    num_workers = num_workers or min(os.cpu_count(), batch_size)
    train_dataset, test_dataset = random_split_dataset(
        full_dataset, train_size, test_size,
        trans_val=TRANSFORM2_VAL if model_dims == 2 else TRANSFORM_VAL
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # Load previous epoch
    model = _KNOWN_MODELS[model_name]().float()
    if from_ckpt is None:
        from_ckpt = _find_latest_ckpt(session_name)
    previous_epoch = _ckpt_epoch(from_ckpt) or 0
    if from_ckpt:
        logging.info(f'Load checkpoint from {from_ckpt!r}.')
        model.load_state_dict(torch.load(from_ckpt, map_location='cpu'))
    else:
        logging.info(f'No checkpoint found, new model will be used.')

    if preference < 0:
        loss_weight = torch.as_tensor([torch.e, 1.0]) ** -preference
    else:
        loss_weight = torch.as_tensor([1.0, torch.e]) ** preference
    loss_fn = FocalLoss(weight=loss_weight).to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    model, optimizer, train_dataloader, test_dataloader, scheduler = \
        accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler)

    for epoch in range(previous_epoch + 1, max_epochs + 1):
        running_loss = 0.0
        train_correct, train_total = 0, 0
        train_fp, train_fn = 0, 0
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.float()
            inputs = inputs.to(accelerator.device)
            labels = labels.to(accelerator.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_fp += (preds[labels == 0] == 1).sum().item()
            train_fn += (preds[labels == 1] == 0).sum().item()
            train_total += labels.shape[0]

            loss = loss_fn(outputs, labels)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            scheduler.step()

        epoch_loss = running_loss / train_total
        # epoch_loss = torch.tensor(epoch_loss).to(accelerator.device)
        # epoch_loss = accelerator.reduce(epoch_loss, reduction="sum")

        train_accuracy = train_correct / train_total
        train_fp_p = train_fp / train_total
        train_fn_p = train_fn / train_total

        if accelerator.is_local_main_process:
            logging.info(f'Epoch [{epoch}/{max_epochs}], loss: {epoch_loss:.6f}, '
                         f'train accuracy: {train_accuracy:.4f}, '
                         f'false positive: {train_fp_p:.4f}, false negative: {train_fn_p:.4f}')
            if writer:
                writer.add_scalar('train/loss', epoch_loss, epoch)
                writer.add_scalar('train/accuracy', train_accuracy, epoch)
                writer.add_scalar('train/fp', train_fp_p, epoch)
                writer.add_scalar('train/fn', train_fn_p, epoch)

        model.eval()
        if epoch % eval_epoch == 0:
            with torch.no_grad():
                test_correct, test_total = 0, 0
                test_fp, test_fn = 0, 0
                for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
                    inputs = inputs.float().to(accelerator.device)
                    labels = labels.to(accelerator.device)

                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    test_correct += (preds == labels).sum().item()
                    test_fp += (preds[labels == 0] == 1).sum().item()
                    test_fn += (preds[labels == 1] == 0).sum().item()
                    test_total += labels.shape[0]

                test_accuracy = test_correct / test_total
                test_fp_p = test_fp / test_total
                test_fn_p = test_fn / test_total

                # test_accuracy = torch.tensor(test_accuracy).to(accelerator.device)
                # test_accuracy = accelerator.reduce(test_accuracy, reduction="sum")

                if accelerator.is_local_main_process:
                    # test_accuracy = test_accuracy.item()
                    logging.info(f'Epoch {epoch}, test accuracy: {test_accuracy:.4f}, '
                                 f'false positive: {test_fp_p:.4f}, false negative: {test_fn_p:.4f}')
                    if writer:
                        writer.add_scalar('test/accuracy', test_accuracy, epoch)
                        writer.add_scalar('test/fp', test_fp_p, epoch)
                        writer.add_scalar('test/fn', test_fn_p, epoch)

        if accelerator.is_local_main_process and epoch % save_per_epoch == 0:
            current_ckpt_file = os.path.join(_CKPT_DIR, f'monochrome-{session_name}-{epoch}.ckpt')
            torch.save(model.state_dict(), current_ckpt_file)
            logging.info(f'Saved to {current_ckpt_file!r}.')
