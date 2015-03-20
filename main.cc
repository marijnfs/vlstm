#include <iostream>
#include <cuda.h>

#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
#include "tiff.h"

using namespace std;

int main() {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);

	//int kg(3), ko(3), c(1);
	int kg(7), ko(7), c(1);

	Volume tiff_data = open_tiff("7nm/input.tif", true);
	// Volume tiff_label = open_tiff("7nm/input.tif", true);
	Volume tiff_label = open_tiff("7nm/binary-labels.tif", false, true);

	cout << tiff_data.shape << endl;
	cout << tiff_label.shape << endl;
	//cout << tif_data.to_vector() << endl;
	//cout << tif_label.to_vector() << endl;

	//VolumeShape shape{100, 1, 256, 256};
	//VolumeShape shape{168, 1, 255, 255};
	VolumeShape shape = tiff_data.shape;
	// Volume in_data(shape);
	// Volume out_data(shape);

	// in_data.init_normal(0, .5);
	// out_data.init_normal(0, .5);

	VolumeNetwork net(shape);
	net.add_vlstm(3, 3, 10);
	net.add_fc(3);
	net.add_tanh();
	net.add_vlstm(3, 3, 3);
	net.add_fc(2);
	//net.add_sigmoid();
	net.add_softmax();
	// net.add_tanh();
	// net.add_tanh();
	// net.add_fc(1);

	net.finish();
	net.init_normal(0, .3);

	net.set_input(tiff_data);
	// net.volumes[0]->x.draw_slice("lala.png", 4);

	while (true) {

		net.forward();
		net.output().draw_slice("lala.png", 4);
		cout << "loss " << net.calculate_loss(tiff_label) << endl;
		net.backward();
		net.update(10.);
	}

	// VLSTM vlstm(shape, kg, ko, c);
	// vlstm.x.x.from_volume(tif_data);
	// vlstm.init_normal(0.0, .05);

	// while (true) {
	// 	vlstm.clear();
	// 	cout << "forward" << endl;
	// 	vlstm.forward();
	// 	cout << "output norm: " << vlstm.y.x.norm() << endl;
	// 	//cout << vlstm.y.x.to_vector() << endl;
	// 	//cout << tif_label.to_vector() << endl;
	// 	// vlstm.operations[4]->volumes["c"]->x.draw_slice("c.png", 4);
	// 	vlstm.y.x.draw_slice("out.png", 8);
	// 	vlstm.x.x.draw_slice("in.png", 8);
	// 	tif_label.draw_slice("target.png", 8);

	// 	vlstm.y.diff.from_volume(tif_label);
	// 	vlstm.y.diff -= vlstm.y.x;

	// 	float norm = vlstm.y.diff.norm();

	// 	cout << "diff norm: " << norm << endl;
	// 	cout << "backward" << endl;
	// 	vlstm.backward();
	// 	vlstm.update(.02);
	// 	cout << norm << endl;
	// }

	cudaDeviceSynchronize();
}
