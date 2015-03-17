#include <iostream>
#include <cuda.h>

#include "volume.h"
#include "vlstm.h"
#include "handler.h"
#include "tiff.h"

using namespace std;

int main() {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);

	int kg(3), ko(3), c(1);

	Volume tif_data = open_tiff("7nm/input.tif");
	Volume tif_label = open_tiff("7nm/binary-labels.tif");

	cout << tif_data.shape << endl;
	cout << tif_label.shape << endl;
	//cout << tif_data.to_vector() << endl;
	//cout << tif_label.to_vector() << endl;

	//VolumeShape shape{100, 1, 256, 256};
	//VolumeShape shape{168, 1, 255, 255};
	VolumeShape shape = tif_data.shape;
	// Volume in_data(shape);
	// Volume out_data(shape);

	// in_data.init_normal(0, .5);
	// out_data.init_normal(0, .5);

	VLSTM vlstm(shape, kg, ko, c);
	vlstm.x.x.from_volume(tif_data);
	vlstm.init_normal(0, .05);

	while (true) {
		vlstm.clear();
		cout << "forward" << endl;
		vlstm.forward();
		cout << "output norm: " << vlstm.y.x.norm() << endl;
		//cout << vlstm.y.x.to_vector() << endl;
		//cout << tif_label.to_vector() << endl;
		vlstm.y.x.draw_slice("out.png", 4);
		vlstm.x.x.draw_slice("in.png", 4);
		tif_label.draw_slice("target.png", 4);
	
		vlstm.y.diff.from_volume(tif_label);
		vlstm.y.diff -= vlstm.y.x;

		float norm = vlstm.y.diff.norm();

		cout << "diff norm: " << norm << endl;
		cout << "backward" << endl;
		vlstm.backward();
		vlstm.update(.0000006);
		cout << norm << endl;
	}

	cudaDeviceSynchronize();
}
