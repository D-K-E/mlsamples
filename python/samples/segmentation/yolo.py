"""
\brief yolo based instance segmenter
"""
from ultralytics import YOLO
from .segment_interface import SegmentationMask, Segmenter
from pathlib import Path
from collections.abc import Iterator


def load_model(path: str = "YOLOv8x-seg.pt"):
    "load a segmentation model based on YoloV8"
    model = YOLO(name)
    return model


class YoloMask(SegmentationMask):
    """"""

    def __init__(self, result):
        super().__init__(masks=result.masks, frame=result.orig_frame)


class YoloSegmenter(Segmenter):
    """"""

    def __init__(self):
        self.model = load_model()

    def segment(self, video: Path) -> Iterator[SegmentationMask]:
        """"""
        results = self.model(str(video), device="cuda", stream=True)
        for result in results:
            mask = YoloMask(result)
            yield mask
