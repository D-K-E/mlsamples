"""
\brief detectron based instance segmenter
"""
from pathlib import Path
from mlsamples.task.detection.detect_interface import Detection, Detector
from mlsamples.misc.utils import load_detectron
from collections.abc import Iterator
from torchvision.io import read_video


class DetectronDetection(Detection):
    """"""

    def __init__(self, result: dict):
        super().__init__(frame=result["frame"], boxes=result["pred_boxes"])


class DetectronDetector(Segmenter):
    """"""

    def __init__(self):
        self.model = load_detectron()

    def detect(self, video: Path) -> Iterator[Detection]:
        """"""
        clip = read_video(str(video))
        T, Height, Width, Channel = clip.shape
        for t in range(T):
            frame = clip[t, :]
            preds = self.model(frame)
            result = dict(pred_boxes=preds["instances"].pred_boxes, frame=frame)
            d = DetectronDetection(result)
            yield d
