import os.path

from ultralytics import YOLO

from ..base import _TRAIN_DIR as _GLOBAL_TRAIN_DIR

_TRAIN_DIR = os.path.join(_GLOBAL_TRAIN_DIR, 'hand_detect')


def train(train_cfg: str, session_name: str, level: str = 's',
          max_epochs: int = 200, **kwargs):
    # Load a pretrained YOLO model (recommended for training)
    _last_pt = os.path.join(_TRAIN_DIR, session_name, 'weights', 'last.pt')
    if os.path.exists(_last_pt):
        model, resume = YOLO(_last_pt), True
    else:
        model, resume = YOLO(f'yolov8{level}.pt'), False

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    model.train(
        data=train_cfg, epochs=max_epochs,
        name=session_name, project=_TRAIN_DIR,
        save=True, plots=True,
        exist_ok=True, resume=resume,
        **kwargs
    )
