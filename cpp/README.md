# Building on Linux

## Dependencies

- OpenCV 4.5+
- libtorch
- C++17 compiler
- CMake 3.18+

Rest of the dependencies are downloaded with FetchContent.

## Building

- `mkdir build`
- `cd build`
- `cmake ..`
- `make`

These commands should create two executables: `min.out` and `run.out`.

Run `min.out` to test whether `torch` is linked correctly or not.


## Tested Toolchain

```
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0

-- Found Torch: ./mlsamples/cpp/lib/libtorch/lib/libtorch.so  
-- Found OpenCV: (found version "4.5.4") 
```

# Usage

Once you built the program, you launch the `run.out` to see available options:

- `./run.out -h`: This should give you something like the following:

```
Usage: ai task runner [--help] [--version] [--backend VAR] --task VAR --video VAR --save_name VAR

Optional arguments:
  -h, --help     shows help message and exits 
  -v, --version  prints version information and exits 
  --backend      [nargs=0..1] [default: "yolo"]
  --task         [nargs=0..1] [default: "segment"]
  --video        path to input video [required]
  --save_name    save name for output video, e.g. out.mp4 [nargs=0..1] [default: "out.mp4"]
```
There is one backend (`yolo`) and two tasks (`segment`, `detect`) that
are available. To launch `segment` task with `yolo` backend on a video:

- `$ ./run.out --task segment --video ../../data/out02.mp4 --save_name segment03.mp4`
