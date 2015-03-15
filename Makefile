all:
	nvcc -std=c++11 -O3 --compiler-options "" -DNDEBUG -L. -Iinc -arch compute_30 -I/usr/local/cuda-6.5/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options main.cc simulator.cc tree.cc config.cc -o main
