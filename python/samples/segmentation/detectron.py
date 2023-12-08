"""
\brief detectron based instance segmenter
"""
from .segment_interface import SegmentationMask, Segmenter
from pathlib import Path
from collections.abc import Iterator
from torchvision.io import read_video

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def load_model(path: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    "load a segmentation model based on Detectron2"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(path)
    model = DefaultPredictor(cfg)
    return model


class DetectronMask(SegmentationMask):
    """"""

    def __init__(self, result: dict):
        super().__init__(masks=result["pred_masks"], frame=result["frame"])


class DetectronSegmenter(Segmenter):
    """"""

    def __init__(self):
        self.model = load_model()

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
