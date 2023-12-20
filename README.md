# DL Samples

These samples should help you get started in DL based projects.
The main purpose in both samples is to isolate the model code from the pipeline
that would use its services. 
They serve as a toy deployment example in a desktop environment. 

There are two samples in this repository: `python` and `c++`. Essentially they
are both simple AI based video analysis tools. They both support multiple
tasks:

- python: detection, segmentation, pose estimation
- c++: detection, segmentation

Python version supports multiple backends:

- YOLOv8
- Detectron2

C++ version supports only YOLOv8.

Python version automatically downloads needed models, but C++ version requires
them to be present in the `assets` directory.

Though C++ and Python is using the same backend model `YOLOv8`, they produce
different outputs. This is due to preprocessing and post processing functions
involved prior and after the inference. The python version uses `ultralytics`
library to leverage preprocessing/postprocessing. The C++ version implements
it from scratch with some help from various open sourced libraries.

Each repository contains its proper build instructions. We are also providing
test and reference videos in case the user wants to not only inspect the
source code but also launch the program and see what it provides.
These videos can be found in the `data` folder.

All samples are tested with Ubuntu 22.04.
