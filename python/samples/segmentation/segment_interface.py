"""
\brief segmentation interface for all backends
"""
from abc import ABC, abstractmethod
from pathlib import Path

from typing import List
import torch
from collections.abc import Iterator


class BaseMask(ABC):
    """"""

    @property
    @abstractmethod
    def masks(self) -> torch.Tensor:
        """"""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame(self) -> torch.Tensor:
        """"""
        raise NotImplementedError


class SegmentationMask(BaseMask):
    def __init__(self, masks, frame):
        self._masks = masks
        self._frame = frame

    @property
    def masks(self):
        if not isinstance(self._masks, torch.Tensor):
            raise TypeError(f"masks must be a tensor but it is {type(self._masks)}")
        return self._masks

    @property
    def frame(self):
        """"""
        if not isinstance(self._frame, torch.Tensor):
            raise TypeError(f"masks must be a tensor but it is {type(self._frame)}")

        return self._frame


class Segmenter(ABC):
    """"""

    @abstractmethod
    def segment(self, video: Path) -> Iterator[SegmentationMask]:
        """"""
        raise NotImplementedError
