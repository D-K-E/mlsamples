"""
\brief very simple pipeline builder
"""
from pathlib import Path
from mlsamples.engine.engine_interface import BaseEngine
from mlsamples.engine.detection import DetectionEngine
from mlsamples.engine.segmentation import SegmentationEngine
from mlsamples.engine.pose import PoseEstimationEngine
from mlsamples.visual.detection import DetectionVisual
from mlsamples.visual.segmentation import SegmentationVisual
from mlsamples.visual.pose import KeypointsVisual

from mlsamples.visual.visual_interface import BaseVisual
from mlsamples.misc.utils import is_type
from mlsamples.misc.utils import Backend, Task
import torch
from torchvision.io import write_video
import moviepy.editor as mpy


class Pipeline:
    """"""

    def __init__(self, engine: BaseEngine, visualizer: BaseVisual, save_location: Path):
        """"""
        is_type(engine, "engine", BaseEngine, True)
        self.engine = engine

        #
        is_type(visualizer, "visualizer", BaseVisual, True)
        self.visualizer = visualizer

        is_type(save_location, "save_location", Path, True)
        self.out = save_location

    def run(self, video: Path):
        """\brief Run the pipeline and save the result to location"""
        engine_result = self.engine.run(video)
        result = self.visualizer.draw(engine_result)
        clip = mpy.VideoFileClip(str(video))
        fps = clip.fps
        write_video(filename=str(self.out), video_array=result, fps=fps)


def build_pipeline(backend: Backend, task: Task, save_location: Path):
    """"""
    if task == Task.SEGMENTATION:
        engine = SegmentationEngine(backend)
        visual = SegmentationVisual()
    elif task == Task.DETECTION:
        engine = DetectionEngine(backend)
        visual = DetectionVisual()
    elif task == Task.POSE_ESTIMATION:
        engine = PoseEstimationEngine(backend)
        visual = KeypointsVisual()
    else:
        raise ValueError(f"unsupported task {str(task)}")
    pipe = Pipeline(engine=engine, visualizer=visual, save_location=save_location)
    return pipe
