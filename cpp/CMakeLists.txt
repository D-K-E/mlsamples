cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aidev)

find_package(Torch REQUIRED PATHS "lib/libtorch")

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE "RelWithDebInfo")

add_executable(min.out 
    "src/minimal.cpp"
    )

target_link_libraries(min.out "${TORCH_LIBRARIES}")
set_property(TARGET min.out PROPERTY CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)


add_executable(run.out 
  "src/run.cpp"
  "src/mlsamples/loader.cpp"
  "src/mlsamples/pipeline.cpp"
  "src/mlsamples/visual/drawer.cpp"
  "src/mlsamples/task/detection.cpp"
  "src/mlsamples/task/yoloutils.cpp"
  "src/mlsamples/task/segmentation.cpp"
)

target_link_libraries(run.out "${TORCH_LIBRARIES}")
target_link_libraries(run.out argparse)
target_link_libraries(run.out ${OpenCV_LIBRARIES})

target_include_directories(run.out PUBLIC "include")

target_include_directories(run.out PRIVATE "${OpenCV_INCLUDE_DIRS}")

set_property(TARGET run.out PROPERTY CXX_STANDARD 17)
target_compile_definitions(
    run.out
    PRIVATE
    ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
    )
