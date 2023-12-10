"""
\brief detectron based instance segmenter
"""
from pathlib import Path
from mlsamples.task.detection.detect_interface import Detection, Detector
from mlsamples.misc.utils import load_detectron
from mlsamples.misc.utils import Task
from collections.abc import Iterator
from torchvision.io import read_video
import torch
import numpy as np


class DetectronDetection(Detection):
    """"""

    def __init__(self, result: dict):
        frame = result["frame"]
        if isinstance(frame, np.ndarray):
            frame = torch.tensor(frame)

        boxes = []
        pboxes = result["pred_boxes"].tensor
        boxes = torch.zeros_like(pboxes)
        boxes[:, 0] = pboxes[:, 0]
        boxes[:, 1] = pboxes[:, 1]
        boxes[:, 2] = pboxes[:, 2] - pboxes[:, 0]
        boxes[:, 3] = pboxes[:, 3] - pboxes[:, 1]

        super().__init__(frame=frame, boxes=boxes)


class DetectronDetector(Detector):
    """"""

    def __init__(self):
        self.model = load_detectron(Task.DETECTION)

    def detect(self, video: Path) -> Iterator[Detection]:
        """"""
        v = read_video(str(video))
        clip = v[0]
        T, Height, Width, Channel = clip.shape
        for t in range(T):
            frame = clip[t, :].numpy()
            preds = self.model(frame)
            result = dict(pred_boxes=preds["instances"].pred_boxes, frame=frame)
            d = DetectronDetection(result)
            yield d
