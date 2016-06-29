#include <iostream>
#include <cuda.h>
#include <sstream>

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


	vector<string> paths = walk(autism_path, ".nii");

	for (auto path : paths) {
	  cout << path << endl;
	  NiftiVolume nifti_volume(path);
	  Volume vol = nifti_volume.get_volume();
	  vol.draw_slice("test.png", 40);
	  return 0;
	}
	return 0;	  //inputs[n].draw_slice_rgb("input.bmp",10);


	VolumeShape sub_shape{30, 1, 64, 64};
	VolumeNetwork net(sub_shape);

	// net.add_fc(8);
	// net.add_vlstm(7, 7, 8);
	// net.add_fc(8);
	// net.add_tanh();
	// net.add_fc(2);
	// net.add_softmax();


	//Marijn net
	//Wonmin net
	// net.add_fc(16);
	// net.add_vlstm(7, 7, 16);
	// net.add_fc(32);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 32);
	// net.add_fc(64);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 64);
	// net.add_fc(64);
	// net.add_tanh();
	// net.add_fc(5);
	// net.add_softmax();

	//Wonmin net
	//// net.add_fc(16);
	net.add_vlstm(7, 7, 16);
	net.add_fc(32);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(32);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(128);
	net.add_tanh();
	net.add_fc(5);
	net.add_softmax();

	//Wonmin net2
	/*net.add_fc(32);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(64, .1);
	net.add_tanh();
	net.add_vlstm(7, 7, 64);
	net.add_fc(128, .1);
	net.add_tanh();
	net.add_vlstm(7, 7, 128);
	net.add_fc(256, .1);
	net.add_tanh();
	net.add_fc(2);
	net.add_softmax();
	*/


	net.finish();
	if (argc > 1)
	  net.load(argv[1]);
	else
	  net.init_uniform(.1);

	cout << net.volumes[0]->x.shape << endl;

	logger << "begin description\n";
	logger << "subvolume shape " << sub_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";


	//Training Setup
	
	int epoch(0);
	Trainer trainer(net.param_vec.n, .03, .0001, 1000);
	epoch = 0;
	float last_loss = 9999999.;

	while (true) {
	  Timer total_timer;
	  ostringstream ose;
	  //ose << img_path << "mom_sub_in-" << epoch << ".png";
	  //copy_subvolume(inputs[brainnum], net.input(), outputs[brainnum], label_subset, false, rand()%2, false, false); // rotate, xflip, yflip, zflip
	  //copy stuff to network input
	  net.forward();

	  Volume target(VolumeShape{1, 2, 1, 1});
	  float loss = net.calculate_loss(target);
	  logger << "epoch: " << epoch << ": loss " << (loss / sub_shape.size()) << "\n";
	  last_loss = loss;

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

	cudaDeviceSynchronize();
}
