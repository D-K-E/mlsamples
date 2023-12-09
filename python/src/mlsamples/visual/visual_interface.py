"""
\brief visual interface
"""

from abc import ABC, abstractmethod
import torch


class BaseVisual(ABC):
    """"""

    @abstractmethod
    def draw(self, engine_result) -> torch.Tensor:
        """"""
        raise NotImplementedError
