import os.path
import re
from functools import partial
from typing import Optional, Type

import torch
from accelerate import Accelerator
from ditk import logging
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .alexnet import MonochromeAlexNet
from .dataset import MonochromeDataset, random_split_dataset
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
          max_epochs: int = 500, learning_rate: float = 0.001, weight_decay: float = 1e-3,
          num_workers: Optional[int] = None, device: Optional[str] = None,
          save_per_epoch: int = 10, eval_epoch: int = 5, model_name: str = 'alexnet'):
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    session_name = session_name or model_name
    _log_dir = os.path.join(_LOG_DIR, session_name)

    if accelerator.is_local_main_process:
        os.makedirs(_log_dir, exist_ok=True)
        os.makedirs(_CKPT_DIR, exist_ok=True)
        writer = SummaryWriter(_log_dir)
        writer.add_custom_scalars({
            "general": {
                "accuracy": ["Multiline", ["train/accuracy", "test/accuracy"]],
            },
        })
    else:
        writer = None

    # Initialize dataset
    full_dataset = MonochromeDataset(dataset_dir, bins=feature_bins, fc=fc)
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # 使用 random_split 函数拆分数据集
    train_dataset, test_dataset = random_split_dataset(full_dataset, train_size, test_size)
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

    # Try use cude
    # if torch.cuda.is_available():
    #    model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = lambda inputs, targets: sigmoid_focal_loss(inputs, targets, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(model, optimizer,
                                                                                         train_dataloader,
                                                                                         test_dataloader, scheduler)

    for epoch in range(previous_epoch + 1, max_epochs + 1):
        running_loss = 0.0
        train_correct = 0
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.float()
            inputs = inputs.to(accelerator.device)
            labels = labels.to(accelerator.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            train_correct += (torch.argmax(outputs, dim=1) == labels).sum().detach().item()

            loss = loss_fn(outputs, labels)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        epoch_loss = torch.tensor(epoch_loss).to(accelerator.device)
        epoch_loss = accelerator.reduce(epoch_loss, reduction="sum")

        if accelerator.is_local_main_process:
            epoch_loss = epoch_loss.item()
            logging.info(
                f'Epoch [{epoch}/{max_epochs + 1}] loss: {epoch_loss:.4f}, '
                f'with learning rate: {scheduler.get_last_lr()[0]:.6f}'
            )
            if writer:
                writer.add_scalar('train/loss', epoch_loss, epoch)

        train_accuracy = train_correct / len(train_dataset)
        train_accuracy = torch.tensor(train_accuracy).to(accelerator.device)
        train_accuracy = accelerator.reduce(train_accuracy, reduction="sum")

        if accelerator.is_local_main_process:
            train_accuracy = train_accuracy.item()
            logging.info(f'Epoch {epoch} train accuracy: {train_accuracy:.4f}')
            if writer:
                writer.add_scalar('train/accuracy', train_accuracy, epoch)

        if epoch % eval_epoch == 0:
            with torch.no_grad():
                test_correct = 0
                for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
                    inputs = inputs.float()
                    inputs = inputs.to(accelerator.device)
                    labels = labels.to(accelerator.device)

                    outputs = model(inputs)
                    test_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

                test_accuracy = test_correct / len(test_dataset)
                test_accuracy = torch.tensor(test_accuracy).to(accelerator.device)
                test_accuracy = accelerator.reduce(test_accuracy, reduction="sum")

                if accelerator.is_local_main_process:
                    test_accuracy = test_accuracy.item()
                    logging.info(f'Epoch {epoch} test accuracy: {test_accuracy:.4f}')
                    if writer:
                        writer.add_scalar('test/accuracy', test_accuracy, epoch)

        if accelerator.is_local_main_process and epoch % save_per_epoch == 0:
            current_ckpt_file = os.path.join(_CKPT_DIR, f'monochrome-{session_name}-{epoch}.ckpt')
            torch.save(model.state_dict(), current_ckpt_file)
            logging.info(f'Saved to {current_ckpt_file!r}.')
