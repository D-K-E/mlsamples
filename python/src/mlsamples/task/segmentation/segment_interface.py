"""
\brief segmentation interface for all backends
"""
from abc import ABC, abstractmethod
from pathlib import Path

from typing import List
import torch
from collections.abc import Iterator
from mlsamples.misc.utils import FrameContainer


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


class SegmentationMask(BaseMask, FrameContainer):
    def __init__(
        self, masks: Optional[torch.Tensor] = None, frame: Optional[torch.Tensor] = None
    ):
        super().__init__(frame=frame)
        is_optional_type(masks, "masks", torch.Tensor, True)
        self._masks = masks

    def set_masks(self, masks: torch.Tensor):
        """"""
        is_type(masks, "masks", torch.Tensor, True)
        self._masks = masks

    @property
    def masks(self):
        if self._masks is None:
            raise ValueError("masks is none")
        return self._masks


class Segmenter(ABC):
    """"""

    @abstractmethod
    def segment(self, video: Path) -> Iterator[SegmentationMask]:
        """"""
        raise NotImplementedError
