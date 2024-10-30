cmake_minimum_required(VERSION 3.11)
project(CountObj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(/usr/local/cuda/include)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_executable(build
    main11.cpp
    TrtModel.cpp
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    # ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)

target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")
target_link_libraries(build PRIVATE ${OpenCV_LIBS} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})


# cmake_minimum_required(VERSION 3.11)
# project(SSS LANGUAGES CXX)

# add_executable(sss sss.cpp)
