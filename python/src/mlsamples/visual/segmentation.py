"""
\brief segmentation visual
"""
from mlsamples.visual.visual_interface import BaseVisual
from mlsamples.task.segmentation.segment_interface import SegmentationMask
from typing import Iterator
import torch
import cv2
import random
import numpy as np


class SegmentationVisual(BaseVisual):
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

    def draw(self, engine_result: Iterator[SegmentationMask]) -> torch.Tensor:
        """"""
        result = []
        alpha = 0.7
        beta = 1.0 - alpha
        for d in engine_result:
            frame = d.frame
            masks = d.masks
            arr = frame.numpy()
            frame_img = arr.copy()  # cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            temp = np.zeros_like(frame_img)

            for mask in masks:
                color = random.choice(SegmentationVisual.random_colors)
                m = np.array(mask)
                temp = cv2.fillConvexPoly(temp, m, color)
                #
            frame_img = cv2.addWeighted(frame_img, alpha, temp, beta, 0.0)
            #
            # frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            result.append(frame_img)
        #
        return torch.tensor(result)
