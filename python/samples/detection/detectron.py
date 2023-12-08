"""
\brief detectron based instance segmenter
"""
from pathlib import Path
from .detect_interface import Detection, Detector
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


class DetectronDetection(Detection):
    """"""

    def __init__(self, result: dict):
        super().__init__(frame=result["frame"], boxes=result["pred_boxes"])


class DetectronDetector(Segmenter):
    """"""

    def __init__(self):
        self.model = load_model()

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
