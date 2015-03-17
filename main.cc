#include <iostream>
#include <cuda.h>

#include "volume.h"
#include "vlstm.h"

using namespace std;

int main() {
	//VolumeShape shape{100, 1, 512, 512};
	VolumeShape shape{100, 1, 256, 256};
	int kg(3), ko(3), c(1);

	Volume in_data(shape);
	Volume out_data(shape);

	in_data.init_normal(0, .1);
	out_data.init_normal(0, .1);

	VLSTM vlstm(shape, kg, ko, c);
	vlstm.x.x.from_volume(in_data);
	vlstm.init_normal(0, .1);

	while (true) {
		vlstm.clear();
		cout << "forward" << endl;
		cout << vlstm.y.x.norm() << endl;
		cout << vlstm.x.x.norm() << endl;
		cout << vlstm.operations[0]->volumes["x"]->x.norm() << endl;
		vlstm.forward();
		cout << vlstm.y.x.norm() << endl;
		vlstm.y.diff.from_volume(vlstm.y.x);
		vlstm.y.diff -= out_data;
		float norm = vlstm.y.diff.norm();

		cout << "norm: " << norm << endl;
		cout << "backward" << endl;
		vlstm.backward();
		vlstm.update(.01);
		cout << norm << endl;
	}

	cudaDeviceSynchronize();
}
