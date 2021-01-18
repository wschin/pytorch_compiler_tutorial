cmake_minimum_required(VERSION 3.7)

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

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${PYTORCH_DIR}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
pybind11_add_module(pointwise_compiler SHARED ${COMPILER_SRCS})
target_link_libraries(pointwise_compiler PUBLIC torch pybind11 asmjit ${TORCH_PYTHON_LIBRARY})


target_include_directories(pointwise_compiler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PYTORCH_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
    asmjit/src
)

