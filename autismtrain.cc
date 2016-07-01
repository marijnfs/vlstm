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
#include "read_niftii.h"

using namespace std;



int main(int argc, char **argv) {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);


	string autism_path = "/home/marijnfs/data/autism/autistic";
	string control_path = "/home/marijnfs/data/autism/controls";

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



	vector<NiftiVolume> volumes;
	vector<string> paths;
	vector<bool> has_autism;
	for (auto p : walk(autism_path, ".nii")) {	  
	  volumes.push_back(NiftiVolume(p));
	  paths.push_back(p);
	  has_autism.push_back(true);
	}

	for (auto p : walk(control_path, ".nii")) {	  
	  volumes.push_back(NiftiVolume(p));
	  paths.push_back(p);
	  has_autism.push_back(false);
	}

	
	NiftiVolume nifti_volume(paths[0]);
	Volume input_vol = nifti_volume.get_volume();
	input_vol.add_normal(0, .6);
	string name(200, ' ');
	for (size_t i(0); i < input_vol.shape.z; ++i) {
	  sprintf(&name[0], "test_%i.png", i);
	  input_vol.draw_slice(name, i);
	}

	//VolumeShape sub_shape{30, 1, 64, 64};
	VolumeNetwork net(input_vol.shape);

	net.add_vlstm(7, 7, 6);
	net.add_vlstm(7, 7, 6);
	net.add_fc(2);
	net.add_tanh();
	net.add_classify(2);
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


	//Training Setup
	
	int epoch(0);
	Trainer trainer(net.param_vec.n, .0001, .0000001, 1000);
	epoch = 0;
	float last_loss = 9999999.;

	while (true) {

	  vector<int> indices(volumes.size());
	  for (int i(0); i < indices.size(); ++i)
	    indices[i] = i;
	  random_shuffle(indices.begin(), indices.end());

	  float total_loss(0);
	  for (auto index : indices) {
	    Volume input_volume = volumes[index].get_volume();
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
	    last_loss = loss;
	    total_loss += loss;
	    
	    Timer timer;
	    net.backward();
	    cout << "backward took:" << timer.since() << "\n\n";
	      
	    trainer.update(&net.param_vec, net.grad_vec);
	    cout << "grad: ";
	    print_wide(net.grad_vec.to_vector(), 20); //Slow
	    cout << "param: ";
	    print_wide(net.param_vec.to_vector(), 20); //Slow
	    
	    net.save(netname);
	    
	    ++epoch;
	    cout << "epoch time: " << total_timer.since() <<" lr: " << trainer.lr() << endl;
	  }

	  cout << "avg loss: " << (total_loss / indices.size()) << endl;
	}

	cudaDeviceSynchronize();
}
