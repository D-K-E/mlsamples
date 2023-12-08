"""
\brief detection interface for all backends
"""
from abc import ABC, abstractmethod
from pathlib import Path

from collections.abc import Iterator


class BaseDetection(ABC):
    """"""

    @property
    @abstractmethod
    def bboxes(self) -> torch.Tensor:
        """"""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame(self) -> torch.Tensor:
        """"""
        raise NotImplementedError


class Detection(BaseDetection):
    def __init__(self, boxes: torch.Tensor, frame: torch.Tensor):
        self._bboxes = boxes
        self._frame = frame

    @property
    def bboxes(self):
        if not isinstance(self._bboxes, torch.Tensor):
            raise TypeError(f"bboxes must be a tensor but it is {type(self._bboxes)}")
        return self._bboxes

    @property
    def frame(self):
        """"""
        if not isinstance(self._frame, torch.Tensor):
            raise TypeError(f"masks must be a tensor but it is {type(self._frame)}")

        return self._frame


class Detector(ABC):
    """"""

    @abstractmethod
    def detect(self, video: Path) -> Iterator[Detection]:
        """"""
        raise NotImplementedError
