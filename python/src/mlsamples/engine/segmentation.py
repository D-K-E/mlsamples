"""
\brief a segmentation engine based on given backend
"""

from mlsamples.engine.engine_interface import BaseEngine
from mlsamples.misc.utils import Backend
from mlsamples.misc.utils import is_type
from mlsamples.task.segmentation.yolo import YoloSegmenter
from mlsamples.task.segmentation.detectron import DetectronSegmenter
from mlsamples.task.segmentation.segment_interface import SegmentationMask
from typing import Iterator
from pathlib import Path


class SegmentationEngine(BaseEngine):
    """"""

    available_backends = {
        Backend.YOLO: YoloSegmenter,
        Backend.DETECTRON: DetectronSegmenter,
    }

    def __init__(self, backend: Backend):
        """"""
        is_type(backend, "backend", Backend, True)
        if backend not in SegmentationEngine.available_backends:
            raise ValueError(f"str(backend) is not available")
        self.model = SegmentationEngine.available_backends[backend]()

    def run(self, video: Path) -> Iterator[SegmentationMask]:
        """"""
        return self.model.segment(video)
