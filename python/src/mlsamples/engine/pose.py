"""
\brief a pose estimation engine based on given backend
"""

from mlsamples.engine.engine_interface import BaseEngine
from mlsamples.misc.utils import Backend
from mlsamples.misc.utils import is_type
from mlsamples.task.pose.yolo import YoloPoseEstimator
from mlsamples.task.pose.detectron import DetectronPoseEstimator
from mlsamples.task.pose.pose_interface import Keypoints
from typing import Iterator
from pathlib import Path


class PoseEstimationEngine(BaseEngine):
    """"""

    available_backends = {
        Backend.YOLO: YoloPoseEstimator,
        Backend.DETECTRON: DetectronPoseEstimator,
    }

    def __init__(self, backend: Backend):
        """"""
        is_type(backend, "backend", Backend, True)
        if backend not in PoseEstimationEngine.available_backends:
            raise ValueError(f"str(backend) is not available")
        self.model = PoseEstimationEngine.available_backends[backend]()

    def run(self, video: Path) -> Iterator[Keypoints]:
        """"""
        return self.model.estimate_poses(video)
