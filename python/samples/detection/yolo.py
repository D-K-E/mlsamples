"""
\brief yolo based instance detector
"""
from ultralytics import YOLO
from .detect_interface import Detection, Detector
from pathlib import Path
from collections.abc import Iterator


def load_model(path: str = "YOLOv8x.pt"):
    "load a detection model based on YoloV8"
    model = YOLO(name)
    return model


class YoloDetection(Detection):
    """"""

    def __init__(self, result):
        super().__init__(frame=result.orig_frame, boxes=result.bboxes)


class YoloDetector(Detector):
    """"""

    def __init__(self):
        self.model = load_model()

    def detect(self, video: Path) -> Iterator[Detection]:
        """"""
        results = self.model(str(video), device="cuda", stream=True)
        for result in results:
            d = YoloDetection(result)
            yield d
