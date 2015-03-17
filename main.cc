#include <iostream>
#include <cuda.h>

#include "volume.h"
#include "vlstm.h"
#include "handler.h"

using namespace std;

int main() {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);

	//VolumeShape shape{100, 1, 256, 256};
	VolumeShape shape{100, 1, 128, 128};
	int kg(3), ko(3), c(1);

	Volume in_data(shape);
	Volume out_data(shape);

	in_data.init_normal(0, .5);
	out_data.init_normal(0, .5);

	VLSTM vlstm(shape, kg, ko, c);
	vlstm.x.x.from_volume(in_data);
	vlstm.init_normal(0, .05);

	while (true) {
		vlstm.clear();
		cout << "forward" << endl;
		vlstm.forward();
		cout << "output norm: " << vlstm.y.x.norm() << endl;

		vlstm.y.diff.from_volume(vlstm.y.x);
		vlstm.y.diff -= out_data;
		float norm = vlstm.y.diff.norm();

		cout << "diff norm: " << norm << endl;
		cout << "backward" << endl;
		vlstm.backward();
		vlstm.update(-.00001);
		cout << norm << endl;
	}

	cudaDeviceSynchronize();
}
