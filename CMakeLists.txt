# CMakeLists files in this project can
# refer to the root source directory of the project as ${MCDNN_SOURCE_DIR} and

cmake_minimum_required (VERSION 2.8.11)
project (VLSTM)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmakefind/")

#SET(CMAKE_BUILD_TYPE CACHE RELEASE STRING "Build Type")

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
	message(FATAL_ERROR "In-source builds are not permitted. Make a separate folder for building:\nmkdir build; cd build; cmake ..\nBefore that, remove the files already created:\nrm -rf CMakeCache.txt CMakeFiles")
endif()

find_package(CUDA QUIET REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)
find_package(NIFTI REQUIRED)
include_directories(${NIFTI_INCLUDE_DIRS})

find_package(VTK REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})


#find_library(PROTOBUF_LIBRARY protobuf HINTS /usr/lib/x86_64-linux-gnu REQUIRED)
#find_library(PROTOBUF_LIBRARY REQUIRED)
#include_directories(${PROTOBUF_INCLUDE_DIRS})

#include_directories(${CMAKE_CURRENT_BINARY_DIR})


#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "-std=c++11;-O3 -I/home/marijnfs/dev/mcdnn/inc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -I/home/marijnfs/dev/mcdnn/inc")

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -I/home/cvlstm/dev/mcdnn/inc")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -I/home/cvlstm/dev/mcdnn/inc")

SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -L/usr/lib/x86_64-linux-gnu" )

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

cuda_add_executable(autismtrain  ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} autismtrain.cc)
target_link_libraries(autismtrain ${CUDA_curand_LIBRARY} ${PROTOBUF_LIBRARIES} -lleveldb ${OpenCV_LIBS}  -lcudnn -lmcdnn ${NIFTI_LIBRARIES} ${VTK_LIBRARIES})
CUDA_ADD_CUBLAS_TO_TARGET(autismtrain)

cuda_add_executable(autismexamine  ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} autismexamine.cc)
target_link_libraries(autismexamine ${CUDA_curand_LIBRARY} ${PROTOBUF_LIBRARIES} -lleveldb ${OpenCV_LIBS}  -lcudnn -lmcdnn ${NIFTI_LIBRARIES} ${VTK_LIBRARIES})
CUDA_ADD_CUBLAS_TO_TARGET(autismexamine)

cuda_add_executable(bratstrain  ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} bratstrain.cc)
target_link_libraries(bratstrain ${CUDA_curand_LIBRARY} ${PROTOBUF_LIBRARIES} -lleveldb ${OpenCV_LIBS}  -lcudnn -lmcdnn ${NIFTI_LIBRARIES} ${VTK_LIBRARIES})
CUDA_ADD_CUBLAS_TO_TARGET(bratstrain)

#cuda_add_executable(mainbrain  ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} mainbrain.cc)
#target_link_libraries( mainbrain ${CUDA_curand_LIBRARY} ${PROTOBUF_LIBRARIES} -lleveldb ${OpenCV_LIBS}  -lcudnn -lmcdnn )
#CUDA_ADD_CUBLAS_TO_TARGET(mainbrain)


#cuda_add_executable(icub  ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} icub.cc)
#target_link_libraries( icub
#  -lmcdnn ${CUDA_curand_LIBRARY} ${PROTOBUF_LIBRARIES} ${OpenCV_LIBS}  -lleveldb   -lcudnn 
#)
#CUDA_ADD_CUBLAS_TO_TARGET(icub)

#cuda_add_executable(icubeval  ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} icubeval.cc)
#target_link_libraries( icubeval
#	-lmcdnn -lcudnn ${CUDA_curand_LIBRARY} -lleveldb -lopencv_core -lopencv_highgui -lopencv_imgproc
#)
#CUDA_ADD_CUBLAS_TO_TARGET(icubeval)


#cuda_add_executable(evalbrain  ${CPP_HEADERS} ${CPP_SOURCE} ${CU_SOURCE} evalbrain.cc)
#target_link_libraries( evalbrain
#	-lmcdnn -lcudnn ${CUDA_curand_LIBRARY} -lleveldb -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff
#)
#CUDA_ADD_CUBLAS_TO_TARGET(evalbrain)

#install(TARGETS mcdnn
#	LIBRARY DESTINATION /usr/local/lib
#	ARCHIVE DESTINATION /usr/local/lib
#)
