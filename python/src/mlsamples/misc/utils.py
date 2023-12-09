"""
\brief basic utility functions
"""
from typing import Any, Optional
from uuid import uuid4
from enum import Enum, auto

from ultralytics import YOLO
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch


def is_type(field_value, field_name: str, field_type, raise_error: bool = True) -> bool:
    "check type of field value"
    if not isinstance(field_name, str):
        raise TypeError(
            "field_name {0} must be a string but it has type {1}".format(
                str(field_name), str(type(field_name))
            )
        )
    if not isinstance(field_value, field_type):
        if raise_error:
            raise TypeError(
                "field_value {0} must be a {1} but it has type {2}".format(
                    str(field_value), str(field_type), str(type(field_value))
                )
            )
        return False
    return True


def is_optional_type(
    field_value, field_name: str, field_type, raise_error: bool = True
) -> bool:
    "check type of field value"
    if field_value is None:
        return True
    else:
        return is_type(field_value, field_name, field_type, raise_error)


class FrameContainer:
    """"""

    def __init__(self, frame: Optional[torch.Tensor] = None):
        is_optional_type(frame, "frame", torch.Tensor, True)
        self._frame = frame

    def set_frame(self, f: torch.Tensor):
        """"""
        is_type(f, "f", torch.Tensor, True)
        self._frame = f

    @property
    def frame(self) -> torch.Tensor:
        if self._frame is None:
            raise ValueError("frame is none")
        return self._frame


class YoloTask(Enum):
    SEGMENTATION = auto()
    DETECTION = auto()
    POSE_ESTIMATION = auto()


def load_yolo(model_type: YoloTask) -> YOLO:
    """
    Load yolo model
    """
    is_type(model_type, "model_type", YoloModel, True)
    if model_type == YoloTask.SEGMENTATION:
        return YOLO("YOLOv8x-seg.pt")
    elif model_type == YoloTask.DETECTION:
        return YOLO("YOLOv8x.pt")
    elif model_type == YoloTask.POSE_ESTIMATION:
        return YOLO("YOLOv8x-pose.pt")
    else:
        raise ValueError(f"unknown model type {str(model_type)}")


def load_detectron():
    """
    load detectron model
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(path)
    model = DefaultPredictor(cfg)
    return model


class Backend(Enum):
    YOLO = auto()
    DETECTRON = auto()


class Task(Enum):
    SEGMENTATION = auto()
    DETECTION = auto()
    POSE_ESTIMATION = auto()
