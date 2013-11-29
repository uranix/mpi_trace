cmake_minimum_required(VERSION 2.8)

project(newtrace)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(MPI REQUIRED)
find_package(METIS REQUIRED VERSION 5)
find_package(Umfpack REQUIRED)

add_subdirectory(meshProcessor)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${MPI_INCLUDE_PATH})
include_directories(${METIS_INCLUDE_DIR})
include_directories(${UMFPACK_INCLUDES})

add_executable(mesh_prepare mesh_prepare.cpp)
target_link_libraries(mesh_prepare mesh3d ${METIS_LIBRARY})

add_executable(main main.cpp trace.cpp)
target_link_libraries(main mesh3d ${MPI_LIBRARIES} ${UMFPACK_LIBRARIES})
list(APPEND CMAKE_CXX_FLAGS "-std=c++0x")

if (MPI_COMPILE_FLAGS)
	set_target_properties(main PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if (MPI_LINK_FLAGS)
	set_target_properties(main PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()	
