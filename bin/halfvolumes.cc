#include <iostream>
//#include <cuda.h>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>

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


	//string from_path = "/home/marijnfs/data/autism/autistic";
	//string to_path = "/home/marijnfs/data/autism/autistic-halved";
	string from_path = "/home/marijnfs/data/autism/controls";
	string to_path = "/home/marijnfs/data/autism/controls-halved";

	for (auto p : walk(from_path, ".nii")) {
	  string filename = p.substr(p.find_last_of("/")+1);
	  cout << filename << endl;
	  //return 0;
	  NiftiVolume nifti_volume(p);
	  Volume v = nifti_volume.get_volume();
	  //Volume v(p);
	  vector<float> values = v.to_vector();

	  VolumeShape s(v.shape);
	  VolumeShape hf{s.z / 2, s.c, s.w / 2, s.h / 2};

	  vector<float> avg_values(hf.size());
	  
	  for (int z(0); z < hf.z; ++z)
	    for (int c(0); c < hf.c; ++c)
		for (int y(0); y < hf.h; ++y)
		  for (int x(0); x < hf.w; ++x) {
		    float sum(0), N(0);
		    for (int zz(z*2); zz < min(z*2+2, s.z); ++zz)
		      for (int yy(y*2); yy < min(y*2+2, s.h); ++yy)
			for (int xx(x*2); xx < min(x*2+2, s.w); ++xx, ++N)
			  sum += values[s.offset(zz, c, xx, yy)];
		    avg_values[hf.offset(z, c, x, y)] = sum / N;
		  }
	  
	  Volume half_volume(hf);
	  half_volume.from_vector(avg_values);


	  Volume smoothed(s);
	  smooth(half_volume, smoothed, 5, 0);
	  half_volume -= smoothed;
	  vector<Volume*> l{&half_volume, &smoothed};
	  Volume j = join_volumes(l);
	  j.save_file(to_path + "/" + filename + ".vol");
	  
	}
	
}
