"""
\brief keypoints visual
"""
from mlsamples.visual.visual_interface import BaseVisual
from mlsamples.task.pose.pose_interface import Keypoints
from typing import Iterator
import torch
import cv2


class KeypointsVisual(BaseVisual):
    """"""

    def draw(self, engine_result: Iterator[Keypoints]) -> torch.Tensor:
        """"""
        result = []
        for d in engine_result:
            frame = d.frame
            keypoints = d.keypoints.numpy()
            arr = frame.numpy()
            frame_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            temp = np.zeros_like(frame_img)
            for i in range(keypoints.shape[0]):
                x, y = keypoints[i]
                frame_img = cv2.circle(
                    frame_img,  # img
                    (x, y),  # center
                    2,  # radius
                    255,  # white
                    -1,  # thickness
                )
            #
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            result.append(frame_img)
        #
        return torch.tensor(result)
