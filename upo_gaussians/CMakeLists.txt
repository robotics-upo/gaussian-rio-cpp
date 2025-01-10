cmake_minimum_required(VERSION 3.22)

project(upo_gaussians LANGUAGES CUDA CXX)

if (MSVC)
	message(FATAL_ERROR "MSVC is not supported")
endif()

# Maybe in the future
#find_package(Python COMPONENTS Interpreter Development)
#find_package(pybind11 CONFIG REQUIRED)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(PCL 1.3 REQUIRED COMPONENTS common)
find_package(small_gicp REQUIRED)

# PCL uses old-style CMake... we need to wrap it into a proper target
add_library(upo_gaussians_PCL INTERFACE)
target_compile_definitions(upo_gaussians_PCL INTERFACE ${PCL_DEFINITIONS})
target_include_directories(upo_gaussians_PCL INTERFACE ${PCL_INCLUDE_DIRS})
target_link_directories(upo_gaussians_PCL INTERFACE ${PCL_LIBRARY_DIRS})
target_link_libraries(upo_gaussians_PCL INTERFACE ${PCL_LIBRARIES})

find_package(ament_cmake QUIET)
if (ament_cmake_FOUND)
	message(STATUS "We are using ROS 2 to build this")

	set(destdir_LIB     lib)
	set(destdir_INCLUDE include)
	set(destdir_BIN     bin)

elseif (CATKIN_DEVEL_PREFIX OR CATKIN_BUILD_BINARY_PACKAGE)
	message(STATUS "We are using ROS 1 to build this")
	find_package(catkin REQUIRED
		#COMPONENTS ...
	)

	catkin_package(
		INCLUDE_DIRS include
		LIBRARIES upo_gaussians
		#CATKIN_DEPENDS ...
		DEPENDS Eigen3 PCL
	)

	set(destdir_LIB     "${CATKIN_PACKAGE_LIB_DESTINATION}")
	set(destdir_INCLUDE "${CATKIN_GLOBAL_INCLUDE_DESTINATION}")
	set(destdir_BIN     "${CATKIN_GLOBAL_BIN_DESTINATION}")

else()
	message(STATUS "We are using plain CMake to build this")
	include(GNUInstallDirs)

	set(destdir_LIB     "${CMAKE_INSTALL_LIBDIR}")
	set(destdir_INCLUDE "${CMAKE_INSTALL_INCLUDEDIR}")
	set(destdir_BIN     "${CMAKE_INSTALL_BINDIR}")

endif()

add_library(upo_gaussians SHARED)

target_link_libraries(upo_gaussians PUBLIC
	Eigen3::Eigen
	upo_gaussians_PCL
)

target_link_libraries(upo_gaussians PRIVATE
	small_gicp::small_gicp
)

target_compile_options(upo_gaussians PRIVATE
	-Wall
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>
)

target_compile_features(upo_gaussians PUBLIC cxx_std_17)

set_target_properties(upo_gaussians PROPERTIES
	CXX_EXTENSIONS ON

	CUDA_RUNTIME_LIBRARY Shared
	CUDA_STANDARD 17
	#CUDA_ARCHITECTURES "50;53-real;61-real;75-real;86-real"
)

target_include_directories(upo_gaussians PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

target_sources(upo_gaussians PRIVATE
	src/kalman_core.cpp
	src/radar.cpp
	src/rio_base.cpp
	src/rio_ndt.cpp
)

install(
	DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
	DESTINATION ${destdir_INCLUDE}
	FILES_MATCHING PATTERN "*.h*"
)

install(
	TARGETS upo_gaussians upo_gaussians_PCL
	EXPORT upo_gaussiansTargets
	ARCHIVE DESTINATION ${destdir_LIB}
	LIBRARY DESTINATION ${destdir_LIB}
	RUNTIME DESTINATION ${destdir_BIN}
)

if (ament_cmake_FOUND)
	ament_export_targets(upo_gaussiansTargets HAS_LIBRARY_TARGET)
	ament_package()
elseif (NOT catkin_FOUND)
	install(
		EXPORT upo_gaussiansTargets
		DESTINATION ${destdir_LIB}/cmake/upo_gaussians
		FILE upo_gaussiansConfig.cmake
	)
endif()
