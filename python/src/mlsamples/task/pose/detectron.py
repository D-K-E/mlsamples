"""
\brief detectron based pose key points
"""
from mlsamples.task.pose.pose_interface import Keypoints
from mlsamples.task.pose.pose_interface import PoseEstimator
from mlsamples.misc.utils import load_detectron, Task
from mlsamples.misc.utils import is_optional_type
from collections.abc import Iterator
from pathlib import Path
from torchvision.io import read_video
import torch
import numpy as np


class DetectronKeypoints(Keypoints):
    """"""

    def __init__(self, result):
        frame = result["frame"]
        if isinstance(frame, np.ndarray):
            frame = torch.tensor(frame)
        keypoints = []
        for points in result["pred_keypoints"]:
            x = points[0].item()
            y = points[1].item()
            keypoints.append((x, y))

        super().__init__(keypoints=keypoints, frame=frame)


class DetectronPoseEstimator(PoseEstimator):
    """"""

    def __init__(self):
        self.model = load_detectron(Task.POSE_ESTIMATION)

    def estimate_poses(self, video: Path) -> Iterator[Keypoints]:
        """"""
        v = read_video(str(video))
        clip = v[0]
        T, Height, Width, Channel = clip.shape
        for t in range(T):
            frame = clip[t, :].numpy()
            preds = self.model(frame)
            keypoints: torch.Tensor = preds["instances"].pred_keypoints
            # transform keypoints to Tensor[Number of keypoints, XY]
            N, num_kpts, C = keypoints.shape
            nkpts = torch.reshape(keypoints, (N * num_kpts, C))
            kpts = nkpts[:, :2]
            result = dict(pred_keypoints=kpts, frame=frame)
            mask = DetectronKeypoints(result)
            yield mask
