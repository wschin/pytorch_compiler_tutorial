cmake_minimum_required(VERSION 3.7)

set (CMAKE_VERBOSE_MAKEFILE on)

file(GLOB COMPILER_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

# PYTORCH_DIR
IF(DEFINED ENV{PYTORCH_DIR})
  SET(PYTORCH_DIR $ENV{PYTORCH_DIR})
ENDIF()

IF ("${PYTORCH_DIR}" STREQUAL "")
  message(FATAL_ERROR "Please specify the PyTorch directory with -DPYTORCH_DIR=/path/to/pytorch/dir")
ENDIF()

message("Using PyTorch directory ${PYTORCH_DIR}")

link_directories(${PYTORCH_DIR}/lib)

add_subdirectory(pybind11)
add_subdirectory(asmjit)

list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_DIR} /usr/local/cuda-11)
find_package(Torch REQUIRED)

pybind11_add_module(pointwise_compiler SHARED ${COMPILER_SRCS})
target_link_libraries(pointwise_compiler PUBLIC "${TORCH_LIBRARIES}" pybind11 asmjit torch_python)


target_include_directories(pointwise_compiler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PYTORCH_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
    asmjit/src
)

