all:
	nvcc -std=c++11 -O3 --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_52 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu main.cc -o main

icub:
	nvcc -std=c++11 -O3 --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_52 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu icub.cc -o main


brain:
	nvcc -std=c++11 -g --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_52 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu mainbrain.cc -o mainbrain

evalu:
	nvcc -std=c++11 -O3 --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_52 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu eval.cc -o eval

evalbr:
	#nvcc -std=c++11 -O3 --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_52 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu evalbrain.cc -o evalbrain
	nvcc -std=c++11 -g --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_30 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu evalbrain.cc -o evalbrain

grad:
	nvcc -std=c++11 -g --compiler-options "" -DNDEBUG -L../mcdnn -I../mcdnn/inc -Iinc -arch compute_30 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -ltiff -lboost_program_options src/*.cc src/*.cu gradcheck.cc -o gradcheck
