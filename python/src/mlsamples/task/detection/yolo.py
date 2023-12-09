"""
\brief yolo based instance detector
"""
from mlsamples.task.detection.detect_interface import Detection, Detector
from mlsamples.misc.utils import is_optional_type, load_yolo, YoloTask
from pathlib import Path
from collections.abc import Iterator


class YoloDetection(Detection):
    """"""

    def __init__(self, result):
        super().__init__(frame=result.orig_img, boxes=result.bboxes.data)


class YoloDetector(Detector):
    """"""

    def __init__(self):
        ""
        self.model = load_yolo(YoloTask.DETECTION)

    def detect(self, video: Path) -> Iterator[Detection]:
        """"""
        results = self.model(str(video), device="cuda", stream=True)
        for result in results:
            d = YoloDetection(result)
            yield d
