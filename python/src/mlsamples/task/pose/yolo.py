"""
\brief yolo based pose key points
"""
from mlsamples.task.pose.pose_interface import Keypoints
from mlsamples.task.pose.pose_interface import PoseEstimator
from mlsamples.misc.utils import is_optional_type
from mlsamples.misc.utils import YoloTask
from mlsamples.misc.utils import load_yolo
from pathlib import Path
from collections.abc import Iterator
import numpy as np
import torch


class YoloKeypoints(Keypoints):
    """"""

    def __init__(self, result):
        frame = result.orig_img
        if isinstance(frame, np.ndarray):
            frame = torch.tensor(frame)

        keypoints = []
        for points in result.keypoints.xy:
            for point in points:
                x = point[0].item()
                y = point[1].item()
                keypoints.append((x, y))

        super().__init__(keypoints=keypoints, frame=frame)


class YoloPoseEstimator(PoseEstimator):
    """"""

    def __init__(self):
        self.model = load_yolo(YoloTask.POSE_ESTIMATION)

    def estimate_poses(self, video: Path) -> Iterator[Keypoints]:
        """"""
        results = self.model(str(video), device="cpu", stream=True)
        for result in results:
            mask = YoloKeypoints(result)
            yield mask
