cmake_minimum_required(VERSION 3.16)
project(upo_gaussian_tester LANGUAGES CXX)

set(_my_catkin_deps
	roscpp
	rosbag
	pcl_conversions
	upo_gaussians

	std_msgs
	sensor_msgs
)

find_package(Ceres REQUIRED)
find_package(small_gicp REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(catkin REQUIRED COMPONENTS ${_my_catkin_deps})

catkin_package(
	CATKIN_DEPENDS ${_my_catkin_deps}
)

add_library(ugt_common INTERFACE)

target_compile_features(ugt_common INTERFACE cxx_std_17)
target_compile_options(ugt_common INTERFACE
	-Wall
)

target_include_directories(ugt_common INTERFACE
	${catkin_INCLUDE_DIRS}
)

target_link_libraries(ugt_common INTERFACE
	${catkin_LIBRARIES}
	small_gicp::small_gicp
	lz4
	yaml-cpp
)

add_dependencies(ugt_common
	${${PROJECT_NAME}_EXPORTED_TARGETS}
	${catkin_EXPORTED_TARGETS}
)

add_executable(run_offline run_offline.cpp)
target_link_libraries(run_offline PRIVATE ugt_common)

install(
	TARGETS
		run_offline
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
