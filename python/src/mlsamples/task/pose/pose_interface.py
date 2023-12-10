"""
\brief pose estimation interface
"""
from abc import ABC, abstractmethod
from pathlib import Path

from typing import List, Optional, Tuple
import torch
from collections.abc import Iterator
from mlsamples.misc.utils import is_optional_type
from mlsamples.misc.utils import is_type
from mlsamples.misc.utils import FrameContainer


class BaseKeypoints(ABC):
    """"""

    @property
    @abstractmethod
    def keypoints(self) -> List[Tuple[int, int]]:
        """"""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame(self) -> torch.Tensor:
        """"""
        raise NotImplementedError


class Keypoints(FrameContainer, BaseKeypoints):
    def __init__(
        self,
        keypoints: Optional[torch.Tensor] = None,
        frame: Optional[torch.Tensor] = None,
    ):
        super().__init__(frame=frame)
        self._keypoints = keypoints

    def set_keypoints(self, keypoints: List[Tuple[int, int]]):
        """"""
        is_type(keypoints, "keypoints", list, True)
        all(is_type(k, "keypoints", tuple, True) for k in keypoints)
        self._keypoints = keypoints

    @property
    def keypoints(self) -> List[Tuple[int, int]]:
        if self._keypoints is None:
            raise ValueError("keypoints is none")
        return self._keypoints


class PoseEstimator(ABC):
    """"""

    @abstractmethod
    def estimate_poses(self, video: Path) -> Iterator[Keypoints]:
        """"""
        raise NotImplementedError
