"""
\brief detection visual
"""
from mlsamples.visual.visual_interface import BaseVisual
from mlsamples.task.detection.detect_interface import Detection
from typing import Iterator
import torch
import cv2
import random


class DetectionVisual(BaseVisual):
    """"""

    random_colors = [
        [0, 0, 0],  #
        [255, 255, 255],  #
        [255, 0, 0],  #
        [0, 255, 0],  #
        [0, 0, 255],  #
        [255, 255, 0],  #
        [0, 255, 255],  #
        [255, 0, 255],  #
    ]

    def draw(self, engine_result: Iterator[Detection]) -> torch.Tensor:
        """"""
        result = []
        for d in engine_result:
            frame = d.frame
            bboxes = d.bboxes.numpy()
            arr = frame.numpy()
            frame_img = arr.copy()  # cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            rects = []
            for i in range(bboxes.shape[0]):
                bbox = list(map(int, bboxes[i, :].flatten().tolist()))
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]
                rect = (x, y, w, h)
                color = random.choice(DetectionVisual.random_colors)
                frame_img = cv2.rectangle(frame_img, rect, color, 2)
                # rects.append(frame_img)
            #
            # frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            result.append(frame_img)
        #
        return torch.tensor(result)
