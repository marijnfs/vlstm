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
	vector<string> T1 = walk(brain_path, ".mha", "T1.");
	vector<string> T1c = walk(brain_path, ".mha", "T1c.");
	vector<string> T2 = walk(brain_path, ".mha", "T2.");
	vector<string> Flair = walk(brain_path, ".mha", "Flair");

	vector<string> segmentations = walk(brain_path, ".mha", "OT");

	cout << T1.size() << endl;
	cout << T1c.size() << endl;
	cout << T2.size() << endl;
	cout << Flair.size() << endl;
	cout << segmentations.size() << endl;

	for (size_t i(0); i < T1.size(); ++i) {

	  int smooth_std = 8;
	  cout << "processing: " << i << "/" << T1.size() << " :" << T1[i] << endl;

	  Volume seg = read_vtk(segmentations[i], true);
		  
	  Volume input_t1 = read_vtk(T1[i]);
	  Volume smoothed_t1(input_t1.shape);
	  smooth(input_t1, smoothed_t1, smooth_std, 0);
	  input_t1 -= smoothed_t1;

	  Volume input_t1c = read_vtk(T1c[i]);
	  Volume smoothed_t1c(input_t1c.shape);
	  smooth(input_t1c, smoothed_t1c, smooth_std, 0);
	  input_t1c -= smoothed_t1c;

	  Volume input_t2 = read_vtk(T2[i]);
	  Volume smoothed_t2(input_t2.shape);
	  smooth(input_t2, smoothed_t2, smooth_std, 0);
	  input_t2 -= smoothed_t2;

	  Volume input_flair = read_vtk(Flair[i]);
	  Volume smoothed_flair(input_flair.shape);
	  smooth(input_flair, smoothed_flair, smooth_std, 0);
	  input_flair -= smoothed_flair;



	  
	  cout << "vol shape: " << input_t1.shape << endl;
	  //input_t1.draw_volume("test_%i.png");
	  
	  //smoothed_t1.draw_volume("smooth_%i.png");

	  vector<Volume*> volume_list{&input_t1, &smoothed_t1,
	      &input_t1c, &smoothed_t1c,
	      &input_t2, &smoothed_t2,
	      &input_flair, &smoothed_flair};
	  
	  Volume joined = join_volumes(volume_list);

	  cout << joined.shape << endl;
	  ostringstream joined_filename, seg_filename;
	  joined_filename << "joined_" << i << ".vol";
	  seg_filename << "seg_" << i << ".vol";
	  joined.save_file(joined_filename.str());
	  seg.save_file(seg_filename.str());
	}

}
