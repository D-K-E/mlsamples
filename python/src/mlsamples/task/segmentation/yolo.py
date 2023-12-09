"""
\brief yolo based instance segmenter
"""
from mlsamples.task.segmentation.segment_interface import SegmentationMask, Segmenter
from pathlib import Path
from collections.abc import Iterator

from mlsamples.misc.utils import YoloTask
from mlsamples.misc.utils import load_yolo


class YoloMask(SegmentationMask):
    """"""

    def __init__(self, result):
        super().__init__(masks=result.masks, frame=result.orig_img)


class YoloSegmenter(Segmenter):
    """"""

    def __init__(self):
        self.model = load_yolo(YoloTask.SEGMENTATION)

    def segment(self, video: Path) -> Iterator[SegmentationMask]:
        """"""
        results = self.model(str(video), device="cuda", stream=True)
        for result in results:
            mask = YoloMask(result)
            yield mask
