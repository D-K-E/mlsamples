"""
\brief detection visual
"""
from mlsamples.visual.visual_interface import BaseVisual
from mlsamples.task.detection.detect_interface import Detection
from typing import Iterator
import torch
import cv2


class DetectionVisual(BaseVisual):
    """"""

    def draw(self, engine_result: Iterator[Detection]) -> torch.Tensor:
        """"""
        result = []
        for d in engine_result:
            frame = d.frame
            bboxes = d.bboxes.numpy()
            arr = frame.numpy()
            frame_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, :].flatten()
                x, y, w, h = bbox
                rect = (x, y, w, h)
                frame_img = cv2.rectangle(frame_img, rect, 0, 2)
            #
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            result.append(frame_img)
        #
        return torch.tensor(result)
