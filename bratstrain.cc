#include <iostream>
//#include <cuda.h>
#include <sstream>
#include <algorithm>

#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
//#include "tiff.h"
#include "log.h"
#include "divide.h"
#include "raw.h"
#include "trainer.h"
#include "walk.h"
#include "read_vtk.h"

using namespace std;



int main(int argc, char **argv) {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);

	string brain_path = "/home/marijnfs/data/BRATS2015_Training/processed";

	string ds = date_string();
	ostringstream oss;
	oss << "log/result-" << ds << ".log";
	Log logger(oss.str());

	ostringstream oss1;
	oss1 << "log/network-" << ds << ".net";
	string netname = oss1.str();

	vector<string> input_paths = walk(brain_path, ".vol", "joined");
 	vector<string> target_paths = walk(brain_path, ".vol", "seg");
	
	assert(input_paths.size() == target_paths.size());
	//Net parameters
	int kg(7), ko(7), c(1);

	Volume first_input(input_paths[0]);
	
	// Volume tiff_data = open_tiff("7nm/input.tif", true);
	// // Volume outputs[brainnum] = open_tiff("7nm/input.tif", true);
	// Volume tiff_label = open_tiff("7nm/binary-labels.tif", false, true);


	

	//VolumeShape sub_shape{30, 1, 64, 64};

	VolumeNetwork net(first_input.shape);

	net.add_fc(6);

	//net.add_vlstm(7, 7, 4);
	net.add_hwv(7);
	net.add_hwv(7);
	//net.add_hwv(7, 8);
	//net.add_hwv(7, 8);
	//net.add_hwv(7, 8);
	//	net.add_hwv(7, 8);
	//net.add_vlstm(7, 7, 4);
	//net.add_vlstm(7, 7, 4);
	net.add_fc(5);
	net.add_softmax();
	net.finish();

	if (argc > 1)
	  net.load(argv[1]);
	else
	  net.init_normal(0.0, 0.01);

	logger << "begin description\n";
	//logger << "subvolume shape " << sub_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";

	Trainer trainer(net.param_vec.n, 0.0001, 0.000001, 200, .8);

	int iteration(0);
	Volume input, target;

	while (true) {
	  vector<int> indices(input_paths.size());
	  for (size_t i(0); i < indices.size(); ++i) indices[i] = i;
	  random_shuffle(indices.begin(), indices.end());

	  for (int index : indices) {
	    Timer timer;
	    cout << "iteration " << iteration << ", vol: " << input_paths[index] << endl;

	    input.load_file(input_paths[index]);
	    target.load_file(target_paths[index]);
 
	    net.input().from_volume(input);
	    net.forward();
	    net.output().draw_volume("output_%i.png", 2);
	    float loss = net.calculate_loss(target);
	    cout << "loss: " << loss << endl;

	    net.backward();
	    cout << "grad: ";
	    print_wide(net.grad_vec.to_vector(), 20); //Slow
	    cout << "param: ";
	    print_wide(net.param_vec.to_vector(), 20); //Slow
	    
	    trainer.update(&net.param_vec, net.grad_vec);

	    ++iteration;
	    cout << "Took: " << timer.since() << " seconds" << endl;
	    net.save(netname);
	  }
	}

}
