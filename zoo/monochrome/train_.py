import os.path
import re
from functools import partial
from typing import Optional, Type

import torch
from ditk import logging
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .alexnet import MonochromeAlexNet
from .dataset import MonochromeDataset
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .transformer import SigTransformer
from ..base import _TRAIN_DIR as _GLOBAL_TRAIN_DIR

_TRAIN_DIR = os.path.join(_GLOBAL_TRAIN_DIR, 'monochrome')
_LOG_DIR = os.path.join(_TRAIN_DIR, 'logs')
_CKPT_DIR = os.path.join(_TRAIN_DIR, 'ckpts')

_CKPT_PATTERN = re.compile(r'^monochrome-(?P<name>[a-zA-Z\d_\-]+)-(?P<epoch>\d+)\.ckpt$')

_KNOWN_MODELS = {}


def _register_model(cls: Type[nn.Module], *args, name=None, **kwargs):
    name = name or cls.__model_name__
    _KNOWN_MODELS[name] = partial(cls, *args, **kwargs)


_register_model(MonochromeAlexNet)
_register_model(ResNet18)
_register_model(ResNet34)
_register_model(ResNet50)
_register_model(ResNet101)
_register_model(ResNet152)
_register_model(SigTransformer)


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
          num_workers: Optional[int] = None, device: Optional[str] = None,
          save_per_epoch: int = 10, model_name: str = 'alexnet'):
    session_name = session_name or model_name
    _log_dir = os.path.join(_LOG_DIR, session_name)
    os.makedirs(_log_dir, exist_ok=True)
    os.makedirs(_CKPT_DIR, exist_ok=True)
    writer = SummaryWriter(_log_dir)
    writer.add_custom_scalars({
        "general": {
            "accuracy": ["Multiline", ["train/accuracy", "test/accuracy"]],
            "false": ["Multiline", ["train/fp", "train/fn", "test/fp", "test/fn"]],
        },
    })

    # Initialize dataset
    full_dataset = MonochromeDataset(dataset_dir, bins=feature_bins, fc=fc)
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # 使用 random_split 函数拆分数据集
    num_workers = num_workers or min(os.cpu_count(), batch_size)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
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

    # Try use cuda
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if preference < 0:
        loss_weight = torch.as_tensor([torch.e, 1.0]) ** -preference
    else:
        loss_weight = torch.as_tensor([1.0, torch.e]) ** preference
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    for epoch in range(previous_epoch + 1, max_epochs + 1):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        logging.info(f'Epoch [{epoch}/{max_epochs + 1}] loss: {epoch_loss:.4f}, '
                     f'with learning rate: {scheduler.get_last_lr()[0]:.6f}')
        writer.add_scalar('train/loss', epoch_loss, epoch)

        with torch.no_grad():
            train_correct, train_total = 0, 0
            train_fp, train_fn = 0, 0
            for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
                inputs = inputs.float().to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == labels).sum().item()
                train_fp += (preds[labels == 0] == 1).sum().item()
                train_fn += (preds[labels == 1] == 0).sum().item()
                train_total += labels.shape[0]

            train_accuracy = train_correct / train_total
            train_fp_p = train_fp / train_total
            train_fn_p = train_fn / train_total
            logging.info(f'Epoch {epoch}, train accuracy: {train_accuracy:.4f}, '
                         f'false positive: {train_fp_p:.4f}, false negative: {train_fn_p:.4f}')
            writer.add_scalar('train/accuracy', train_accuracy, epoch)
            writer.add_scalar('train/fp', train_fp_p, epoch)
            writer.add_scalar('train/fn', train_fn_p, epoch)

            test_correct, test_total = 0, 0
            test_fp, test_fn = 0, 0
            for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
                inputs = inputs.float().to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                test_correct += (preds == labels).sum().item()
                test_fp += (preds[labels == 0] == 1).sum().item()
                test_fn += (preds[labels == 1] == 0).sum().item()
                test_total += labels.shape[0]

            test_accuracy = test_correct / test_total
            test_fp_p = test_fp / test_total
            test_fn_p = test_fn / test_total
            logging.info(f'Epoch {epoch}, test accuracy: {test_accuracy:.4f}, '
                         f'false positive: {test_fp_p:.4f}, false negative: {test_fn_p:.4f}')
            writer.add_scalar('test/accuracy', test_accuracy, epoch)
            writer.add_scalar('test/fp', test_fp_p, epoch)
            writer.add_scalar('test/fn', test_fn_p, epoch)

        if epoch % save_per_epoch == 0:
            current_ckpt_file = os.path.join(_CKPT_DIR, f'monochrome-{session_name}-{epoch}.ckpt')
            torch.save(model.state_dict(), current_ckpt_file)
            logging.info(f'Saved to {current_ckpt_file!r}.')
