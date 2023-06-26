import os
import random
import re
from typing import Optional

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from ditk import logging
from hbutils.random import global_seed
from sklearn import svm
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import AUROC, AveragePrecision
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from .dataset import TRAIN_TRANSFORM, CCIPImagesDataset, FastCharacterDataset, TEST_TRANSFORM, char_collect_fn
from .loss import MLCELoss
from .model import CCIP
from ..base import _TRAIN_DIR as _GLOBAL_TRAIN_DIR

_TRAIN_DIR = os.path.join(_GLOBAL_TRAIN_DIR, 'ccip')
_LOG_DIR = os.path.join(_TRAIN_DIR, 'logs')
_CKPT_DIR = os.path.join(_TRAIN_DIR, 'ckpts')

_CKPT_PATTERN = re.compile(r'^ccip-(?P<name>[a-zA-Z\d_\-]+)-(?P<epoch>\d+)\.ckpt$')


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


def _sample_analysis(poss, negs, svm_samples: int = 10000):
    poss_cnt, negs_cnt = poss.shape[0], negs.shape[0]
    total = poss_cnt + negs_cnt
    if total > svm_samples:
        s_poss = poss[random.sample(range(poss_cnt), k=int(round(poss_cnt * svm_samples / total)))]
        s_negs = negs[random.sample(range(negs_cnt), k=int(round(negs_cnt * svm_samples / total)))]
    else:
        s_poss, s_negs = poss, negs

    s_poss, s_negs = s_poss.cpu(), s_negs.cpu()
    features = torch.cat([s_poss, s_negs]).detach().numpy()
    labels = torch.cat([torch.ones_like(s_poss), -torch.ones_like(s_negs)]).detach().numpy()

    model = svm.SVC(kernel='linear')  # 线性核
    model.fit(features.reshape(-1, 1), labels)
    predictions = model.predict(features.reshape(-1, 1))

    return poss.mean().item(), poss.std().item(), negs.mean().item(), negs.std().item(), accuracy_score(labels,
                                                                                                        predictions)


def train(dataset_dir: str, session_name: Optional[str] = None, from_ckpt: Optional[str] = None,
          train_ratio: float = 0.8, max_epochs: int = 500, group_size: int = 30,
          learning_rate: float = 0.001, weight_decay: float = 1e-2, tau: float = 0.15,
          save_per_epoch: int = 10, eval_epoch: int = 5, loss_log_iter: int = 20, log_iter: int = 500, num_workers=8,
          model_name: str = 'clip/ViT-B/32', seed: Optional[int] = 0):
    if seed is not None:
        # native random, numpy, torch and faker's seeds are includes
        # if you need to register more library for seeding, see:
        # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
        global_seed(seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    session_name = session_name or re.sub(r'\W+', '-', model_name)
    _log_dir = os.path.join(_LOG_DIR, session_name)

    if accelerator.is_local_main_process:
        os.makedirs(_log_dir, exist_ok=True)
        os.makedirs(_CKPT_DIR, exist_ok=True)
        writer = SummaryWriter(_log_dir)
        writer.add_custom_scalars({
            "contrastive": {
                "train": ["Multiline", ["train/pos/mean", "train/neg/mean"]],
                "test": ["Multiline", ["test/pos/mean", "test/neg/mean"]],
            },
        })
    else:
        writer = None

    model = CCIP(model_name)
    image_dataset = CCIPImagesDataset(dataset_dir)
    train_image_dataset, test_image_dataset = image_dataset.split_dataset(
        test_prob=1 - train_ratio,
        train_transform=Compose(TRAIN_TRANSFORM + model.preprocess),
        test_transform=Compose(TEST_TRANSFORM + model.preprocess),
    )

    train_dataset = FastCharacterDataset(train_image_dataset, group_size, force_prob=False)
    test_dataset = FastCharacterDataset(test_image_dataset, group_size)
    train_dataset.reset()
    test_dataset.reset()
    train_dataloader = DataLoader(train_dataset, batch_size=group_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=char_collect_fn,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=group_size, num_workers=num_workers,
                                 collate_fn=char_collect_fn)

    if from_ckpt is None:
        from_ckpt = _find_latest_ckpt(session_name)
    previous_epoch = _ckpt_epoch(from_ckpt) or 0
    if from_ckpt:
        logging.info(f'Load checkpoint from {from_ckpt!r}.')
        model.load_state_dict(torch.load(from_ckpt, map_location='cpu'))
    else:
        logging.info(f'No checkpoint found, new model will be used.')

    loss_fn = MLCELoss().to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader)//accelerator.num_processes, epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )
    model = torch.compile(model)

    model, optimizer, train_dataloader, test_dataloader, scheduler = \
        accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler)

    metric_auroc = AUROC(task="binary")
    metric_ap = AveragePrecision(task="binary")

    for epoch in range(previous_epoch + 1, max_epochs + 1):
        running_loss = 0.0
        train_pos_total = 0
        pred_list, gt_list = [], []
        model.train()
        num_iter = len(train_dataloader)
        train_dataloader.dataset.reset()
        for i, (inputs, char_ids) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(accelerator.device)  # BxCxHxW
            char_ids = char_ids.to(accelerator.device)  # B

            outputs = model(inputs)  # BxB
            labels = char_ids

            loss = loss_fn(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item() * len(char_ids)
            train_pos_total += len(char_ids)

            mask = torch.ones_like(outputs).bool().cpu()
            mask ^= torch.diag_embed(torch.diag(mask))
            outputs = outputs.detach().cpu()
            gt_same = (char_ids.view(-1, 1) == char_ids.view(1, -1)).detach().cpu()
            pred_list.append(outputs[mask])
            gt_list.append(gt_same.long()[mask])

            with torch.no_grad():
                if (i + 1) % loss_log_iter == 0:
                    mean_loss = running_loss / train_pos_total
                    if writer:
                        writer.add_scalar('train/loss', mean_loss, (epoch - 1) * num_iter + i)
                        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], (epoch - 1) * num_iter + i)
                    running_loss = 0.
                    train_pos_total = 0

                if (i + 1) % log_iter == 0:
                    pred_t = torch.cat(pred_list).to(accelerator.device)
                    gt_t = torch.cat(gt_list).to(accelerator.device)
                    pred_t, gt_t = accelerator.gather_for_metrics((pred_t, gt_t))
                    if accelerator.is_local_main_process:
                        auc = metric_auroc(pred_t, gt_t).item()
                        ap = metric_ap(pred_t, gt_t).item()
                        logging.info(
                            f'Epoch [{epoch}/{max_epochs}]<{i + 1}/{num_iter}>, loss: {mean_loss:.6f}, AUC: {auc:.3e}, AP: {ap:.3e}.')
                        if writer:
                            # writer.add_scalar('train/loss', mean_loss, epoch)
                            writer.add_scalar('train/auc', auc, (epoch - 1) * num_iter + i)
                            writer.add_scalar('train/ap', ap, (epoch - 1) * num_iter + i)

                        pred_list.clear()
                        gt_list.clear()

        model.eval()
        if epoch % eval_epoch == 0:
            with torch.no_grad():
                pred_list, gt_list = [], []
                for i, (inputs, char_ids) in enumerate(tqdm(test_dataloader)):
                    inputs = inputs.to(accelerator.device)  # BxCxHxW
                    char_ids = char_ids.to(accelerator.device)  # B

                    outputs = model(inputs)  # BxB

                    mask = torch.ones_like(outputs).bool().cpu()
                    mask ^= torch.diag_embed(torch.diag(mask))
                    outputs = outputs.detach().cpu()
                    gt_same = (char_ids.view(-1, 1) == char_ids.view(1, -1)).detach().cpu()
                    pred_list.append(outputs[mask])
                    gt_list.append(gt_same.long()[mask])

                pred_t = torch.cat(pred_list).to(accelerator.device)
                gt_t = torch.cat(gt_list).to(accelerator.device)
                pred_t, gt_t = accelerator.gather_for_metrics((pred_t, gt_t))
                if accelerator.is_local_main_process:
                    auc = metric_auroc(pred_t, gt_t).item()
                    ap = metric_ap(pred_t, gt_t).item()
                    logging.info(f'Epoch [{epoch}/{max_epochs}], AUC: {auc:.3e}, AP: {ap:.3e}.')
                    if writer:
                        writer.add_scalar('test/auc', auc, epoch)
                        writer.add_scalar('test/ap', ap, epoch)

                    pred_list.clear()
                    gt_list.clear()

        if accelerator.is_local_main_process and epoch % save_per_epoch == 0:
            current_ckpt_file = os.path.join(_CKPT_DIR, f'ccip-{session_name}-{epoch}.ckpt')
            torch.save(model.state_dict(), current_ckpt_file)
            logging.info(f'Saved to {current_ckpt_file!r}.')
