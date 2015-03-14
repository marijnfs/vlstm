all:
	nvcc -std=c++11 -O3 --compiler-options "" -DNDEBUG -L. -L$(HOME)/software/fuego-1.1/go/ -L$(HOME)/software/fuego-1.1/smartgame/ -L$(HOME)/software/fuego-1.1/simpleplayers/ -Iinc -arch compute_30 -I/usr/local/cuda-6.5/include/ -I$(HOME)/software/fuego-1.1/go/ -I$(HOME)/software/fuego-1.1/smartgame/ -I$(HOME)/software/fuego-1.1/simpleplayers/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options -lfuego_go -lfuego_smartgame -lfuego_simpleplayers main.cc simulator.cc tree.cc config.cc -o main

train:
	nvcc -std=c++11 -O3 -DNDEBUG -L. -L$(HOME)/software/fuego-1.1/go/ -L$(HOME)/software/fuego-1.1/smartgame/ -L$(HOME)/software/fuego-1.1/simpleplayers/ -Iinc -arch compute_30 -I/usr/local/cuda-6.5/include/ -I$(HOME)/software/fuego-1.1/go/ -I$(HOME)/software/fuego-1.1/smartgame/ -I$(HOME)/software/fuego-1.1/simpleplayers/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options -lfuego_go -lfuego_smartgame -lfuego_simpleplayers trainer.cc simulator.cc tree.cc -o trainer

convert:
	nvcc -std=c++11 -O3 -DNDEBUG -L. -L$(HOME)/software/fuego-1.1/go/ -L$(HOME)/software/fuego-1.1/smartgame/ -L$(HOME)/software/fuego-1.1/simpleplayers/ -Iinc -arch compute_30 -I/usr/local/cuda-6.5/include/ -I$(HOME)/software/fuego-1.1/go/ -I$(HOME)/software/fuego-1.1/smartgame/ -I$(HOME)/software/fuego-1.1/simpleplayers/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options -lfuego_go -lfuego_smartgame -lfuego_simpleplayers convert.cc simulator.cc tree.cc -o convert

debug:
	nvcc -std=c++11 -g -DNDEBUG -L. -L$(HOME)/software/fuego-1.1/go/ -L$(HOME)/software/fuego-1.1/smartgame/ -L$(HOME)/software/fuego-1.1/simpleplayers/ -Iinc -arch compute_30 -I/usr/local/cuda-6.5/include/ -I$(HOME)/software/fuego-1.1/go/ -I$(HOME)/software/fuego-1.1/smartgame/ -I$(HOME)/software/fuego-1.1/simpleplayers/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options -lfuego_go -lfuego_smartgame -lfuego_simpleplayers *.cc -o main

sample:
	rm -rf samples2
	cp -r samples samples2

bak:
	rm -rf samples.bak
	cp -r samples samples.bak
