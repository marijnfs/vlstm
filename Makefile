all:
	nvcc -std=c++11 -O3 --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_30 -I/usr/local/cuda-6.5/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options src/*.cc src/*.cu main.cc -o main
