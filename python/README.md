# ML Samples

Sample code for various computer vision related deep learning tasks.

## Build Instructions for Ubuntu 22.04

- Create a virtual environment: `python -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Install the package: `pip install .`
 
You should be able to launch the `run.py` under `src` directory now.

## Usage example

To access the options from the command line interface, launch the `run.py`
from project directory (directory containing `pyproject.toml` file). For
example: 

- Activate virtual environment: `source venv/bin/activate`
- Run the program to see options: `python src/mlsamples/run.py -h`
```
usage: Simple ai assisted video analysis toolkit [-h] --backend {yolo,detectron} --task {segment,detect,pose} --video VIDEO
                                                 [--save_name SAVE_NAME]

options:
  -h, --help            show this help message and exit
  --backend {yolo,detectron}
  --task {segment,detect,pose}
  --video VIDEO         path to input video e.g. face.mp4
  --save_name SAVE_NAME
                        output name for analyzed video e.g. output.mp4
```

A choose a backend and a task and pass the path of a video directory:
```
$ python src/mlsamples/run.py --backend detectron --task segment --video ../data/out02.mp4 --save_name detectron_segment.mp4
```
This would produce a `detectron_segment.mp4` file inside the project
directory.
