"""
\brief detectron based instance segmenter
"""
from mlsamples.task.segmentation.segment_interface import SegmentationMask, Segmenter
from mlsamples.misc.utils import load_detectron
from pathlib import Path
from collections.abc import Iterator
from torchvision.io import read_video
import numpy as np


class DetectronMask(SegmentationMask):
    """"""

    def __init__(self, result: dict):
        frame = result["frame"]
        if isinstance(frame, np.ndarray):
            frame = torch.tensor(frame)
        masks = []
        pmask = result["pred_masks"]
        N, H, W = pmask.shape
        for n in range(N):
            mask = pmask[n, :]
            points = mask.tolist()
        super().__init__(masks=result["pred_masks"], frame=result["frame"])


class DetectronSegmenter(Segmenter):
    """"""

    def __init__(self):
        self.model = load_detectron()

    def segment(self, video: Path) -> Iterator[SegmentationMask]:
        """"""
        clip = read_video(str(video))
        T, Height, Width, Channel = clip.shape
        for t in range(T):
            frame = clip[t, :]
            preds = self.model(frame)
            result = dict(pred_masks=preds["instances"].pred_masks, frame=frame)
            mask = DetectronMask(result)
            yield mask
