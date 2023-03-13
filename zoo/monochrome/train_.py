import os.path
import re
from typing import Optional

import torch
from ditk import logging
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .alexnet import MonochromeAlexNet
from .dataset import MonochromeDataset
from ..base import _TRAIN_DIR as _GLOBAL_TRAIN_DIR

_TRAIN_DIR = os.path.join(_GLOBAL_TRAIN_DIR, 'monochrome')
_LOG_DIR = os.path.join(_TRAIN_DIR, 'logs')
_CKPT_DIR = os.path.join(_TRAIN_DIR, 'ckpts')

_CKPT_PATTERN = re.compile(r'^monochrome-(?P<epoch>\d+)\.ckpt$')


def _find_latest_ckpt() -> Optional[str]:
    if os.path.exists(_CKPT_DIR):
        ckpts = []
        for filename in os.listdir(_CKPT_DIR):
            matching = _CKPT_PATTERN.fullmatch(os.path.basename(filename))
            if matching:
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


def train(dataset_dir: str, from_ckpt: Optional[str] = None, train_ratio: float = 0.8,
          batch_size: int = 4, feature_bins: int = 400, max_epochs: int = 500):
    os.makedirs(_LOG_DIR, exist_ok=True)
    os.makedirs(_CKPT_DIR, exist_ok=True)
    writer = SummaryWriter(_LOG_DIR)

    # Initialize dataset
    full_dataset = MonochromeDataset(dataset_dir, bins=feature_bins)
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # 使用 random_split 函数拆分数据集
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Load previous epoch
    model = MonochromeAlexNet().float()
    if from_ckpt is None:
        from_ckpt = _find_latest_ckpt()
    previous_epoch = _ckpt_epoch(from_ckpt) or 0
    if from_ckpt:
        logging.info(f'Load checkpoint from {from_ckpt!r}.')
        model.load_state_dict(torch.load(from_ckpt))
    else:
        logging.info(f'No checkpoint found, new model will be used.')

    # Try use cude
    if torch.cuda.is_available():
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
        amsgrad=False
    )

    for epoch in range(previous_epoch + 1, max_epochs + 1):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        logging.info(f'Epoch {epoch} loss: {epoch_loss:.4f}')
        writer.add_scalar('train/loss', epoch_loss, epoch)

        current_ckpt_file = os.path.join(_CKPT_DIR, f'monochrome-{epoch}.ckpt')
        torch.save(model.state_dict(), current_ckpt_file)
        logging.info(f'Saved to {current_ckpt_file!r}.')

        with torch.no_grad():
            train_correct = 0
            for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
                inputs = inputs.float()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                train_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

            train_accuracy = train_correct / len(train_dataset)
            logging.info(f'Epoch {epoch} train accuracy: {train_accuracy:.4f}')
            writer.add_scalar('train/accuracy', train_accuracy, epoch)

            test_correct = 0
            for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
                inputs = inputs.float()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                test_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

            test_accuracy = test_correct / len(test_dataset)
            logging.info(f'Epoch {epoch} test accuracy: {test_accuracy:.4f}')
            writer.add_scalar('test/accuracy', test_accuracy, epoch)
