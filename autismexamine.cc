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

	//	input_vol.draw_volume("test_%i.png", 0);

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
	  throw Err("Need a net as argument");
	
	logger << "begin description\n";
	//logger << "subvolume shape " << sub_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";


	//Analysis setup
	for (int index(10); index < volumes.size(); ++index) {
	    Volume input_volume = volumes[index].get_volume();
	    net.input().from_volume(input_volume);

	    {
	      string name(200, ' ');
	      Volume &v(net.input());
	      for (size_t i(0); i < v.shape.z; ++i) {
		sprintf(&name[0], "original_%i.png", i);
		v.draw_slice(name, i);
	      }
	    }
	    
	    //Create Target
	    Volume target(VolumeShape{1, 2, 1, 1});
	    vector<float> autism({1,0}), control({0,1});
	    if (has_autism[index])
	      target.from_vector(autism);
	    else
	      target.from_vector(control);
	    
	    Trainer adjuster(net.input().size(), -.3, -.03, 20, .8);
	    for (int step(0); step < 40; ++step) {
	      //Run Network
	      net.forward();

	      //net.volumes[2]->x.draw_volume("act_0_%i.png", 0);
	      //net.volumes[2]->x.draw_volume("act_1_%i.png", 1);
	      float loss = net.calculate_loss(target);
	      cout << "output: " << net.output().to_vector() << endl;
	      cout << "target: " << target.to_vector() << endl;
	      net.backward();

	      net.volumes[1]->diff.draw_volume("diff_1_0_%i.png", 0);
	      net.volumes[1]->diff.draw_volume("diff_1_1_%i.png", 1);
	      net.volumes[1]->diff.draw_volume("diff_1_2_%i.png", 2);
	      net.volumes[1]->diff.draw_volume("diff_1_3_%i.png", 3);

	      net.volumes[2]->diff.draw_volume("diff_2_0_%i.png", 0);
	      net.volumes[2]->diff.draw_volume("diff_2_1_%i.png", 1);
	      net.volumes[2]->diff.draw_volume("diff_2_2_%i.png", 2);
	      net.volumes[2]->diff.draw_volume("diff_2_3_%i.png", 3);

	      return 0;

	      cout << "grad: "; print_wide(net.volumes[0]->diff.to_vector(), 20);
	      cout << "input: "; print_wide(net.volumes[0]->x.to_vector(), 20);
	      adjuster.update(net.volumes[0]->x.buf, *net.volumes[0]->diff.buf);
	      //*(net.volumes[0]->diff.buf) *= 300;
	      //net.volumes[0]->x -= net.volumes[0]->diff;
	    }

	    
	    net.volumes[0]->x.draw_volume("adjusted_%i.png", 0);
	      
	    
	    return 0;
	}

	cudaDeviceSynchronize();
}
