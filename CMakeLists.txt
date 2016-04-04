# CMakeLists files in this project can
# refer to the root source directory of the project as ${MCDNN_SOURCE_DIR} and

cmake_minimum_required (VERSION 2.8.11)
project (VLSTM)
#SET(CMAKE_BUILD_TYPE CACHE RELEASE STRING "Build Type")

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
	message(FATAL_ERROR "In-source builds are not permitted. Make a separate folder for building:\nmkdir build; cd build; cmake ..\nBefore that, remove the files already created:\nrm -rf CMakeCache.txt CMakeFiles")
endif()

find_package(CUDA QUIET REQUIRED)

#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "-std=c++11;-O3 -I/home/cvlstm/dev/mcdnn/inc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -I/home/cvlstm/dev/mcdnn/inc")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -I/home/cvlstm/dev/mcdnn/inc")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -I/home/cvlstm/dev/mcdnn/inc")

# Specify include directories
include_directories(
    inc
)

#add_subdirectory (inc)
#add_subdirectory (src)

file(GLOB CPP_SOURCE src/*.cc)
file(GLOB CU_SOURCE src/*.cu)
file(GLOB CPP_HEADERS inc/*.h)

#set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_FLAGS -O3)
#set_property(SOURCE ${CU_SOURCE} PROPERTY COMPILE_FLAGS -O3)
#set_property(SOURCE ${CPP_HEADERS} PROPERTY COMPILE_FLAGS -O3)

cuda_add_executable(icub  ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} icub.cc)
target_link_libraries( icub
	-lmcdnn -lcudnn ${CUDA_curand_LIBRARY} -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options
)
CUDA_ADD_CUBLAS_TO_TARGET(icub)

cuda_add_executable(mainbrain  ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} mainbrain.cc)
target_link_libraries( mainbrain
	-lmcdnn -lcudnn ${CUDA_curand_LIBRARY} -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options
)
CUDA_ADD_CUBLAS_TO_TARGET(mainbrain)

#install(TARGETS mcdnn
#	LIBRARY DESTINATION /usr/local/lib
#	ARCHIVE DESTINATION /usr/local/lib
#)