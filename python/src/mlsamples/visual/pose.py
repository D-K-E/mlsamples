"""
\brief keypoints visual
"""
from mlsamples.visual.visual_interface import BaseVisual
from mlsamples.task.pose.pose_interface import Keypoints
from typing import Iterator
import torch
import cv2
import numpy as np
import random


class KeypointsVisual(BaseVisual):
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

    def draw(self, engine_result: Iterator[Keypoints]) -> torch.Tensor:
        """"""
        result = []
        for d in engine_result:
            frame = d.frame
            keypoints = d.keypoints
            arr = frame.numpy()
            frame_img = arr.copy()  # cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            temp = np.zeros_like(frame_img)
            color = random.choice(KeypointsVisual.random_colors)
            for point in keypoints:
                x, y = list(map(int, point))
                frame_img = cv2.circle(
                    frame_img,  # img
                    (x, y),  # center
                    2,  # radius
                    color,  # white
                    -1,  # thickness
                )
            #
            # frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            result.append(frame_img)
        #
        return torch.tensor(result)
