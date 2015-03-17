#include <cuda.h>

#include "volume.h"
#include "vlstm.h"


int main() {
	//VolumeShape shape{100, 1, 512, 512};
	VolumeShape shape{100, 1, 256, 256};
	int kg(3), ko(3), c(1);

	VolumeSet in_data(shape);
	VolumeSet out_data(shape);
	in_data.x.init_normal(0, .1);
	out_data.x.init_normal(0, .1);

	out_data.diff.init_normal(0, .1);
	
	VLSTM vlstm(shape, kg, ko, c);
	vlstm.x.x.init_normal(0, .1);
	vlstm.x.diff.init_normal(0, .1);
	vlstm.init_normal(0, .1);

	vlstm.forward();
	vlstm.backward();
	
	cudaDeviceSynchronize();
}
