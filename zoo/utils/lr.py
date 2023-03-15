from typing import Tuple, List, Union, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

LRTyping = Union[float, List[Union[Tuple[int, float], float]]]


def _process_lr(lr: LRTyping) -> List[Tuple[Optional[int], float]]:
    if isinstance(lr, float):
        return _process_lr([lr])

    sorted_items = sorted([
        (
            0 if isinstance(item, tuple) else 1,
            item[0] if isinstance(item, tuple) else -1,
            item[1] if isinstance(item, tuple) else item,
            i,
        )
        for i, item in enumerate(lr)
    ], key=lambda x: (x[0], x[1], -x[2], x[3]))
    return [(epoch if epoch >= 0 else None, _lr) for _, epoch, _lr, _ in sorted_items]


def get_init_lr(lr: LRTyping) -> float:
    if not lr:
        raise ValueError(f'Unrecognizable lr - {lr!r}.')
    lr = _process_lr(lr)
    _, _first_lr = lr[0]
    return _first_lr


def get_dynamic_lr_scheduler(optimizer: Optimizer, lr: LRTyping, **kwargs) -> LambdaLR:
    if not lr:
        raise ValueError(f'Unrecognizable lr - {lr!r}.')
    lr = _process_lr(lr)
    _, _first_lr = lr[0]

    def _epoch_to_lambda(epoch: int):
        for _ep, _lr_value in lr:
            if _ep is None or epoch <= _ep:
                return _lr_value / _first_lr
        else:
            _ep, _lr_value = lr[-1]
            return _lr_value / _first_lr

    return LambdaLR(optimizer, lr_lambda=_epoch_to_lambda, **kwargs)
