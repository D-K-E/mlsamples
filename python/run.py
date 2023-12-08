"""
\brief main entrance point of command line program
"""
import argparse
from datetime import datetime
from pathlib import Path
from samples.taskrunner import Backend, Task


def run_task(backend, task, video: Path):
    """"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simple ai assisted video analysis toolkit")
    backends = dict(yolo=Backend.YOLO, detectron=Backend.DETECTRON)
    tasks = dict(
        segment=Task.SEGMENTATION,
        detect=Task.OBJECT_DETECTION,
    )

    parser.add_argument(
        "--backend", required=True, choices=list(backends.keys()), default="yolo"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(tasks.keys()),
        default="detect",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="path to input video e.g. face.mp4",
    )
    now = datetime.now(datetime.UTC)
    ts = "-".join(map(str, now.timetuple()))
    parser.add_argument(
        "--save_name",
        help="output name for analyzed video e.g. output.mp4",
        default="out" + ts + ".mp4",
    )
    args = parser.parse_args()

    backend = backends[args.backend]
    task = tasks[args.task]
    in_video = Path(args.video)
    if not in_video.exists():
        raise FileNotFoundError(f"{args.video} does not exists")
