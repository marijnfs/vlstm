all:
	nvcc -std=c++11 -g --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_30 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu main.cc -o main
	#nvcc -std=c++11 -O3 --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_30 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options src/*.cc src/*.cu main.cc -o main

evalu:
	nvcc -std=c++11 -g --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_30 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu eval.cc -o eval

grad:
	nvcc -std=c++11 -g --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_30 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu gradcheck.cc -o gradcheck
