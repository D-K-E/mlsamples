"""
\brief detection interface for all backends
"""
from abc import ABC, abstractmethod
from pathlib import Path
from mlsamples.misc.utils import is_optional_type
from mlsamples.misc.utils import is_type
from mlsamples.misc.utils import FrameContainer

from collections.abc import Iterator
import torch
import numpy as np


class BaseDetection(ABC):
    """"""

    @property
    @abstractmethod
    def bboxes(self) -> torch.Tensor:
        """"""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame(self) -> np.ndarray:
        """"""
        raise NotImplementedError


class Detection(FrameContainer, BaseDetection):
    def __init__(self, boxes: torch.Tensor, frame: np.ndarray):
        super().__init__(frame=frame)
        is_optional_type(boxes, "boxes", torch.Tensor, True)
        self._bboxes = boxes

    def set_bboxes(self, bboxes: torch.Tensor):
        """"""
        is_type(bboxes, "bboxes", torch.Tensor, True)
        self._bboxes = bboxes

    @property
    def bboxes(self):
        """"""
        if self._bboxes is None:
            raise ValueError("bboxes is none")
        return self._bboxes


class Detector(ABC):
    """"""

    @abstractmethod
    def detect(self, video: Path) -> Iterator[Detection]:
        """"""
        raise NotImplementedError
