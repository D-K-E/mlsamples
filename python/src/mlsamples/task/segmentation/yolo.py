"""
\brief yolo based instance segmenter
"""
from mlsamples.task.segmentation.segment_interface import SegmentationMask, Segmenter
from pathlib import Path
from collections.abc import Iterator

from mlsamples.misc.utils import YoloTask
from mlsamples.misc.utils import load_yolo
import numpy as np
import torch


class YoloMask(SegmentationMask):
    """"""

    def __init__(self, result):
        frame = result.orig_img
        if isinstance(frame, np.ndarray):
            frame = torch.tensor(frame)
        points = result.masks.xy
        masks = []
        for point_arr in points:
            ps = []
            for point in point_arr:
                x = int(point[0])
                y = int(point[1])
                ps.append((x, y))
            masks.append(ps)
        super().__init__(masks=masks, frame=frame)


class YoloSegmenter(Segmenter):
    """"""

    def __init__(self):
        self.model = load_yolo(YoloTask.SEGMENTATION)

    def segment(self, video: Path) -> Iterator[SegmentationMask]:
        """"""
        results = self.model(str(video), device="cpu", stream=True)
        for result in results:
            mask = YoloMask(result)
            yield mask
