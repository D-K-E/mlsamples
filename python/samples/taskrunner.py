"""
Task runner for samples
"""
from enum import Enum, auto


class Backend(Enum):
    YOLO = auto()
    DETECTRON = auto()


class Task(Enum):
    SEGMENTATION = auto()
    ACTION_RECOGNITION = auto()
    OBJECT_DETECTION = auto()

class EngineBuilder:
