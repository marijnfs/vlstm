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


	string autism_path = "/home/marijnfs/data/autism/autistic-halved";
	string control_path = "/home/marijnfs/data/autism/controls-halved";

	vector<string> paths;
	vector<bool> has_autism;
	for (auto p : walk(autism_path, ".nii")) {
	  paths.push_back(p);
	  has_autism.push_back(true);
	}
	//	input_vol.draw_volume("test_%i.png", 0);


	//Choose volume
	int index = 0;
	Volume input_volume(paths[index]);
	
	//Make Network
	VolumeNetwork net(input_volume.shape);

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
	  throw Err("Need a net as argument");
	


	Volume target(VolumeShape{1, 2, 1, 1});
	vector<float> autism({1,0}), control({0,1});
	if (has_autism[index])
	  target.from_vector(autism);
	else
	  target.from_vector(control);
	
	net.input().from_volume(input_volume);
	Trainer adjuster(net.input().size(), -.03, -.03, 20, .8);
	for (int step(0); step < 10; ++step) {
	  //Run Network
	  net.forward();
	  
	  //net.volumes[2]->x.draw_volume("act_0_%i.png", 0);
	  //net.volumes[2]->x.draw_volume("act_1_%i.png", 1);
	  float loss = net.calculate_loss(target);
	  cout << "output: " << net.output().to_vector() << endl;
	  cout << "target: " << target.to_vector() << endl;
	  net.backward();
	  /*
	  net.volumes[1]->diff.draw_volume("diff_1_0_%i.png", 0);
	  net.volumes[1]->diff.draw_volume("diff_1_1_%i.png", 1);
	  net.volumes[1]->diff.draw_volume("diff_1_2_%i.png", 2);
	  net.volumes[1]->diff.draw_volume("diff_1_3_%i.png", 3);
	  
	  net.volumes[2]->diff.draw_volume("diff_2_0_%i.png", 0);
	  net.volumes[2]->diff.draw_volume("diff_2_1_%i.png", 1);
	  net.volumes[2]->diff.draw_volume("diff_2_2_%i.png", 2);
	  net.volumes[2]->diff.draw_volume("diff_2_3_%i.png", 3);
	  
	      return 0;
	  */
	  cout << "grad: "; print_wide(net.volumes[0]->diff.to_vector(), 20);
	  cout << "input: "; print_wide(net.volumes[0]->x.to_vector(), 20);
	  adjuster.update(net.volumes[0]->x.buf, *net.volumes[0]->diff.buf);
	  //*(net.volumes[0]->diff.buf) *= 300;
	  //net.volumes[0]->x -= net.volumes[0]->diff;
	}
	
	
	net.volumes[0]->x.draw_volume("adjusted_%i.png", 0);
	
	cudaDeviceSynchronize();	
	return 0;

}
