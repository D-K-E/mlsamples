"""
\brief detectron based instance segmenter
"""
from mlsamples.task.segmentation.segment_interface import SegmentationMask, Segmenter
from mlsamples.misc.utils import load_detectron
from mlsamples.misc.utils import Task
from pathlib import Path
from collections.abc import Iterator
from torchvision.io import read_video
import numpy as np
import torch


class DetectronMask(SegmentationMask):
    """"""

    def __init__(self, result: dict):
        frame = result["frame"]
        if isinstance(frame, np.ndarray):
            frame = torch.tensor(frame)
        pmask = result["pred_masks"]
        pmask_nz = torch.nonzero(pmask)
        masks = [[] for _ in range(pmask.shape[0])]
        for p in pmask_nz:
            n, y, x = p
            masks[n].append((x, y))
        super().__init__(masks=masks, frame=frame)


class DetectronSegmenter(Segmenter):
    """"""

    def __init__(self):
        self.model = load_detectron(Task.SEGMENTATION)

    def segment(self, video: Path) -> Iterator[SegmentationMask]:
        """"""
        v = read_video(str(video))
        clip = v[0]
        T, Height, Width, Channel = clip.shape
        for t in range(T):
            frame = clip[t, :].numpy()
            preds = self.model(frame)
            result = dict(pred_masks=preds["instances"].pred_masks, frame=frame)
            mask = DetectronMask(result)
            yield mask
