"""
\brief yolo based instance detector
"""
from mlsamples.task.detection.detect_interface import Detection, Detector
from mlsamples.misc.utils import is_optional_type, load_yolo, YoloTask
from pathlib import Path
from collections.abc import Iterator
import numpy as np
import torch


class YoloDetection(Detection):
    """"""

    def __init__(self, result):
        frame = result.orig_img
        if isinstance(frame, np.ndarray):
            frame = torch.tensor(frame)
        boxes = result.boxes.xywh
        if isinstance(boxes, np.ndarray):
            boxes = torch.tensor(boxes)
        super().__init__(frame=frame, boxes=boxes)


class YoloDetector(Detector):
    """"""

    def __init__(self):
        """"""
        self.model = load_yolo(YoloTask.DETECTION)

    def detect(self, video: Path) -> Iterator[Detection]:
        """"""
        results = self.model(str(video), device="cpu", stream=True)
        for result in results:
            d = YoloDetection(result)
            yield d
