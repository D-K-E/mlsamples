"""
\brief detectron based pose key points
"""
from mlsamples.task.pose.pose_interface import Keypoints
from mlsamples.task.pose.pose_interface import PoseEstimator
from mlsamples.misc.utils import load_detectron
from mlsamples.misc.utils import is_optional_type
from collections.abc import Iterator
from pathlib import Path
from torchvision.io import read_video


class DetectronKeypoints(Keypoints):
    """"""

    def __init__(self, result):
        super().__init__(keypoints=result["pred_keypoints"], frame=result["frame"])


class DetectronPoseEstimator(PoseEstimator):
    """"""

    def __init__(self):
        self.model = load_detectron()

    def estimate_poses(self, video: Path) -> Iterator[Keypoints]:
        """"""
        clip = read_video(str(video))
        T, Height, Width, Channel = clip.shape
        for t in range(T):
            frame = clip[t, :]
            preds = self.model(frame)
            result = dict(pred_keypoints=preds["instances"].pred_keypoints, frame=frame)
            mask = DetectronKeypoints(result)
            yield mask