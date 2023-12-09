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


class YoloKeypoints(Keypoints):
    """"""

    def __init__(self, result):
        super().__init__(keypoints=result.keypoints.xy, frame=result.orig_img)


class YoloPoseEstimator(PoseEstimator):
    """"""

    def __init__(self):
        self.model = load_yolo(YoloTask.POSE_ESTIMATION)

    def estimate_poses(self, video: Path) -> Iterator[Keypoints]:
        """"""
        results = self.model(str(video), device="cuda", stream=True)
        for result in results:
            mask = YoloKeypoints(result)
            yield mask
