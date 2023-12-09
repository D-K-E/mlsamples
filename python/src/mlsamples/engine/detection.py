"""
\brief a detection engine based on given backend
"""

from mlsamples.engine.engine_interface import BaseEngine
from mlsamples.misc.utils import Backend
from mlsamples.misc.utils import is_type
from mlsamples.task.detection.yolo import YoloDetector
from mlsamples.task.detection.detectron import DetectronDetector
from mlsamples.task.detection.detect_interface import Detection
from typing import Iterator
from pathlib import Path


class DetectionEngine(BaseEngine):
    """"""

    available_backends = {
        Backend.YOLO: YoloDetector,
        Backend.DETECTRON: DetectronDetector,
    }

    def __init__(self, backend: Backend):
        """"""
        is_type(backend, "backend", Backend, True)
        if backend not in DetectionEngine.available_backends:
            raise ValueError(f"str(backend) is not available")
        self.model = DetectionEngine.available_backends[backend]()

    def run(self, video: Path) -> Iterator[Detection]:
        """"""
        return self.model.detect(video)
