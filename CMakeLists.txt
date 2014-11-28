cmake_minimum_required(VERSION 2.8)

project(newtrace)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(MPI REQUIRED)
find_package(METIS REQUIRED VERSION 5)
find_package(Umfpack REQUIRED)
find_package(CUDA REQUIRED)

add_subdirectory(meshProcessor)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${MPI_INCLUDE_PATH})
include_directories(${METIS_INCLUDE_DIR})
include_directories(${UMFPACK_INCLUDES})

list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall -g")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20 -use_fast_math -Xptxas -v -g -G")
set(CUDA_VERBOSE_BUILD ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF) # Otherwise -std=c++11 will be passed to -Xcompiler
set(CUDA_HOST_COMPILER g++)

cuda_add_library(kernels STATIC gpu.cu kernels.cu)

add_executable(mesh_prepare mesh_prepare.cpp)
target_link_libraries(mesh_prepare mesh3d ${METIS_LIBRARY})

add_executable(main main.cpp umfsolve.cpp LebedevQuad.cpp MeshView.cpp)
target_link_libraries(main mesh3d kernels ${MPI_LIBRARIES} ${UMFPACK_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

if (MPI_COMPILE_FLAGS)
	set_target_properties(main PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if (MPI_LINK_FLAGS)
	set_target_properties(main PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
