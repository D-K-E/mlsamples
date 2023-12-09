"""
\brief base engine interface
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator


class BaseEngine(ABC):
    """"""

    @abstractmethod
    def run(self, video: Path) -> Iterator:
        raise NotImplementedError
