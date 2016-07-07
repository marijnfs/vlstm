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


	//string brain_path = "/home/marijnfs/data/BRATS2015_Training/HGG/brats_tcia_pat118_0001/VSD.Brain_3more.XX.O.OT.42293/VSD.Brain_3more.XX.O.OT.42293.mha";
	//string brain_path = "/home/marijnfs/data/BRATS2015_Training/HGG/brats_tcia_pat149_0001/VSD.Brain.XX.O.MR_Flair.35595/VSD.Brain.XX.O.MR_Flair.35595.mha";
	string brain_path = "/home/marijnfs/data/BRATS2015_Training/HGG";
	for (auto p : walk(brain_path, ".mha", "Flair")) {
	  cout << p << endl;
	  Volume input_vol = read_vtk(p);
	  cout << "vol shape: " << input_vol.shape << endl;
	  input_vol.draw_volume("test_%i.png");
	  return 0;
	}
	return 0;
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


	
	return 0;
	//VolumeShape sub_shape{30, 1, 64, 64};
	/*
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
	*/


}
