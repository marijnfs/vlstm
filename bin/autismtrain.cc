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

using namespace std;



int main(int argc, char **argv) {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);


	string autism_path = "/home/marijnfs/data/autism/autistic-halved";
	string control_path = "/home/marijnfs/data/autism/controls-halved";

	string ds = date_string();
	ostringstream oss;
	oss << "log/result-" << ds << ".log";
	Log logger(oss.str());

	ostringstream oss1;
	oss1 << "log/network-" << ds << ".net";
	string netname = oss1.str();

	//Net parameters
	int kg(7), ko(7), c(1);

	// Volume tiff_data = open_tiff("7nm/input.tif", true);
	// // Volume outputs[brainnum] = open_tiff("7nm/input.tif", true);
	// Volume tiff_label = open_tiff("7nm/binary-labels.tif", false, true);



	vector<Volume*> volumes;
	vector<string> paths;
	vector<bool> has_autism;
	for (auto p : walk(autism_path, ".vol")) {
	  volumes.push_back(new Volume(p));
	  //volumes[volumes.size()-1].load_file(p);
	  paths.push_back(p);
	  has_autism.push_back(true);
	}

	for (auto p : walk(control_path, ".vol")) {
	  volumes.push_back(new Volume(p));
	  paths.push_back(p);
	  has_autism.push_back(false);
	}

	
	Volume &input_vol = *volumes[0];
	input_vol.draw_volume("in_%i.png");
	//VolumeShape sub_shape{30, 1, 64, 64};
	VolumeNetwork net(input_vol.shape);

	net.add_vlstm(7, 7, 32);
	net.add_fc(32);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(2);
	net.add_tanh();
	net.add_classify(2);
	net.add_softmax();
	net.finish();

	if (argc > 1)
	  net.load(argv[1]);
	else
	  net.init_normal(0.0, 0.03);

	logger << "begin description\n";
	//logger << "subvolume shape " << sub_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";


	//Training Setup
	
	int epoch(0);
	Trainer trainer(net.param_vec.n, .0001, .0000001, 200, .5);
	epoch = 0;
	float last_avg_loss = 9999999.;

	while (true) {

	  vector<int> indices(volumes.size());
	  for (int i(0); i < indices.size(); ++i)
	    indices[i] = i;
	  random_shuffle(indices.begin(), indices.end());

	  float total_loss(0);
	  for (auto index : indices) {
	    //index = 0;
	    Volume &input_volume = *volumes[index];
	    float noise(.5);
	    input_volume.add_normal(.0, noise); //
	    net.input().from_volume(input_volume);
	    Timer total_timer;
	    //ostringstream ose;
	    //ose << img_path << "mom_sub_in-" << epoch << ".png";
	    //copy_subvolume(inputs[brainnum], net.input(), outputs[brainnum], label_subset, false, rand()%2, false, false); // rotate, xflip, yflip, zflip
	    //copy stuff to network input
	    net.forward();
	    
	    Volume target(VolumeShape{1, 2, 1, 1});
	    
	    vector<float> autism({1,0}), control({0,1});
	    if (has_autism[index])
	      target.from_vector(autism);
	    else
	      target.from_vector(control);
	    float loss = net.calculate_loss(target);
	    cout << "output: " << net.output().to_vector() << endl;
	    cout << "target: " << target.to_vector() << endl;
	    logger << "epoch: " << epoch << ": loss " << loss << "\n";
	    total_loss += loss;
	    
	    Timer timer;
	    net.backward();
	    cout << "backward took:" << timer.since() << "\n\n";
	      
	    trainer.update(&net.param_vec, net.grad_vec);
	    cout << "in: ";
	    print_wide(net.input().to_vector(), 23); //Slow
	    cout << "grad: ";
	    print_wide(net.grad_vec.to_vector(), 23); //Slow
	    cout << "param: ";
	    print_wide(net.param_vec.to_vector(), 23); //Slow
	    
	    ++epoch;
	    cout << "epoch time: " << total_timer.since() <<" lr: " << trainer.lr() << endl;
	  }

	  float avg_loss = total_loss / indices.size();
	  cout << "avg loss: " << avg_loss << endl;
	  logger << "avg loss: " << avg_loss << "\n";

	  if (avg_loss < last_avg_loss)
	    net.save(netname);
	  last_avg_loss = avg_loss;
	}

	cudaDeviceSynchronize();
}
