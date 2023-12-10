"""
\brief segmentation interface for all backends
"""
from abc import ABC, abstractmethod
from pathlib import Path

from typing import List, Optional, Tuple
import torch
from collections.abc import Iterator
from mlsamples.misc.utils import FrameContainer
from mlsamples.misc.utils import is_optional_type


class BaseMask(ABC):
    """"""

    @property
    @abstractmethod
    def masks(self) -> List[List[Tuple[int, int]]]:
        """"""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame(self) -> torch.Tensor:
        """"""
        raise NotImplementedError


class SegmentationMask(FrameContainer, BaseMask):
    def __init__(
        self,
        masks: Optional[List[List[Tuple[int, int]]]] = None,
        frame: Optional[torch.Tensor] = None,
    ):
        super().__init__(frame=frame)
        is_optional_type(masks, "masks", list, True)
        self._masks = masks

    def set_masks(self, masks: List[List[Tuple[int, int]]]):
        """"""
        is_type(masks, "masks", list, True)
        all(is_type(m, "m", list, True) for m in masks)
        self._masks = masks

    @property
    def masks(self):
        """"""
        if self._masks is None:
            raise ValueError("masks is none")
        return self._masks


class Segmenter(ABC):
    """"""

    @abstractmethod
    def segment(self, video: Path) -> Iterator[SegmentationMask]:
        """"""
        raise NotImplementedError
