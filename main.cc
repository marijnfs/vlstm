#include <iostream>
#include <cuda.h>
#include <sstream>

#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
#include "tiff.h"
#include "log.h"

using namespace std;

int main(int argc, char **argv) {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);

	ostringstream oss;
	oss << "log/result-" << date_string() << ".log";
	Log logger(oss.str());

	ostringstream oss1;
	oss1 << "log/network-" << date_string() << ".net";
	string netname = oss1.str();

	//int kg(3), ko(3), c(1);
	int kg(7), ko(7), c(1);

	// Volume tiff_data = open_tiff("7nm/input.tif", true);
	// // Volume tiff_label = open_tiff("7nm/input.tif", true);
	// Volume tiff_label = open_tiff("7nm/binary-labels.tif", false, true);

	Volume tiff_data = open_tiff("isbi/input.tif", true);
	Volume tiff_label = open_tiff("isbi/label.tif", false, true);

	cout << tiff_data.shape << endl;
	cout << tiff_label.shape << endl;
	// return 0;
	//cout << tiff_data.to_vector() << endl;
	//cout << tiff_label.to_vector() << endl;

	//VolumeShape shape{100, 1, 256, 256};
	//VolumeShape shape{168, 1, 255, 255};
	VolumeShape shape = tiff_data.shape;
	// Volume in_data(shape);
	// Volume out_data(shape);

	// in_data.init_normal(0, .5);
	// out_data.init_normal(0, .5);

	VolumeNetwork net(shape);

	net.add_fc(8);
	//net.add_tanh();

	net.add_vlstm(3, 5, 8);
	net.add_fc(8);
	net.add_tanh();
	net.add_vlstm(3, 5, 10);
	net.add_fc(10);
	net.add_tanh();
	// net.add_vlstm(3, 5, 8);
	// net.add_fc(8);
	// net.add_tanh();
	net.add_fc(2);
	//net.add_sigmoid();
	net.add_softmax();
	// net.add_tanh();
	// net.add_tanh();
	// net.add_fc(1);

	net.finish();
	//net.init_normal(0, .1);
	net.init_uniform(.5);

	cout << net.volumes[0]->x.shape << endl;
	cout << tiff_data.shape << endl;
	net.set_input(tiff_data);
	net.volumes[0]->x.draw_slice("in_3.png", 3);
	tiff_label.draw_slice("label_3.png", 3);

	if (argc > 1) {
	  net.load(argv[1]);
	}

	int epoch(0);
	while (true) {

		Timer ftimer;
		net.forward();
		cout << "forward took:" << ftimer.since() << endl;

		ostringstream oss;
		oss << "img/lala_" << epoch << ".png";
		net.output().draw_slice(oss.str(), 3);
		//cout << net.param.to_vector() << endl;
		logger << "epoch: " << epoch << ": loss " << net.calculate_loss(tiff_label) << "\n";


		Timer timer;
		net.backward();
		cout << "backward took:" << timer.since() << "\n\n";

		// cout << net.grad.to_vector() << endl;
		// net.update(.1);

		//SGD
		// net.grad *= .00001;
		// net.param += net.grad;

		//RMS PROP
		float decay = epoch < 4 ? 0.5 : 0.9;
		float eps = .00001;
		//float lr = 0.001;
		//float lr = 0.01;

		float lr = epoch < 4 ? .001 : .01;
		net.a = net.grad;
		net.a *= net.a;
		net.rmse *= decay;
		net.a *= (1.0 - decay);
		net.rmse += net.a;

		net.b = net.rmse;
		net.b.sqrt();
		net.b += eps;

		net.c = net.grad;
		net.c /= net.b;

		//extra trick

		net.d = net.c;
		net.d *= (1.0 - decay);
		net.e *= decay;
		net.e += net.d;

		net.d = net.e;
		net.d.abs();
		net.c *= net.d;


		//update
		net.c.clip(3.);
		net.c *= lr;
		net.param += net.c;

		++epoch;
		net.save(netname);
		// return 0;
	}

	// VLSTM vlstm(shape, kg, ko, c);
	// vlstm.x.x.from_volume(tiff_data);
	// vlstm.init_normal(0.0, .05);

	// while (true) {
	// 	vlstm.clear();
	// 	cout << "forward" << endl;
	// 	vlstm.forward();
	// 	cout << "output norm: " << vlstm.y.x.norm() << endl;
	// 	//cout << vlstm.y.x.to_vector() << endl;
	// 	//cout << tiff_label.to_vector() << endl;
	// 	// vlstm.operations[4]->volumes["c"]->x.draw_slice("c.png", 4);
	// 	vlstm.y.x.draw_slice("out.png", 8);
	// 	vlstm.x.x.draw_slice("in.png", 8);
	// 	tiff_label.draw_slice("target.png", 8);

	// 	vlstm.y.diff.from_volume(tiff_label);
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
