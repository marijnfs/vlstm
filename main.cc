#include <iostream>
#include <cuda.h>
#include <sstream>

#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
#include "tiff.h"
#include "log.h"
#include "divide.h"

using namespace std;

void print_last(vector<float> vals, int n) {
	for (size_t i(vals.size() - n); i < vals.size(); ++i)
		cout << vals[i] << " ";
	cout << endl;
}

int main(int argc, char **argv) {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);


	string ds = date_string();
	ostringstream oss;
	oss << "log/result-" << ds << ".log";
	Log logger(oss.str());

	ostringstream oss1;
	oss1 << "log/network-" << ds << ".net";
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

	// Volume test(VolumeShape{3, 1, 100, 100});
	// Volume test2(VolumeShape{3, 2, 100, 100});
	// copy_subvolume(tiff_data, test, tiff_label, test2);
	// test.draw_slice("img/blabla.png",2);
	// test2.draw_slice("img/blabla_label.png",2);
	// copy_subvolume(tiff_data, test, tiff_label, test2);
	// test.draw_slice("img/blabla2.png",2);
	// test2.draw_slice("img/blabla2_label.png",2);


	// return 0;
	//cout << tiff_data.to_vector() << endl;
	//cout << tiff_label.to_vector() << endl;

	//VolumeShape shape{100, 1, 256, 256};
	//VolumeShape shape{168, 1, 255, 255};
	VolumeShape sub_shape = tiff_data.shape;
	// Volume in_data(shape);
	// Volume out_data(shape);

	// in_data.init_normal(0, .5);
	// out_data.init_normal(0, .5);
	sub_shape.w = 32;
	sub_shape.h = 32;
	sub_shape.z = 8;
	// sub_shape.w = 32;
	// sub_shape.h = 32;
	// sub_shape.z = 16;
	// sub_shape.w = 256;
	// sub_shape.h = 256;
	// sub_shape.z = 8;

	//We need a volume for sub targets
	Volume label_subset(VolumeShape{sub_shape.z, tiff_label.shape.c, sub_shape.w, sub_shape.h});

	VolumeNetwork net(sub_shape);

	// net.add_fc(8);
	// net.add_vlstm(7, 7, 8);
	// net.add_fc(8);
	// net.add_tanh();
	// net.add_fc(2);
	// net.add_softmax();


	//Marijn net
/*	net.add_fc(8);
	net.add_vlstm(7, 7, 32);
	net.add_fc(32);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(32);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(32);
	net.add_tanh();
	net.add_fc(2);
	net.add_softmax();*/

	//Wonmin net
	// net.add_fc(16);
	// net.add_vlstm(7, 7, 16);
	// net.add_fc(25);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 32);
	// net.add_fc(45);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 64);
	// //net.add_fc(32);
	// // net.add_tanh();
	// net.add_fc(2);
	// net.add_softmax();

	//Wonmin net2
	net.add_fc(32);
	net.add_vlstm(7, 7, 32);
	net.add_fc(64, .5);
	net.add_tanh();
	net.add_vlstm(7, 7, 64);
	net.add_fc(128, .5);
	net.add_tanh();
	net.add_vlstm(7, 7, 128);
	net.add_fc(256, .5);
	net.add_tanh();
	net.add_fc(2);
	net.add_softmax();


	net.finish();
	//net.init_normal(0, .1);
	net.init_uniform(.1);


	cout << net.volumes[0]->x.shape << endl;
	cout << tiff_data.shape << endl;
	// net.set_input(tiff_data);
	// net.volumes[0]->x.draw_slice("in_3.png", 3);
	// tiff_label.draw_slice("label_3.png", 3);

	if (argc > 1) {
	  net.load(argv[1]);
	}

	logger << "begin description\n";
	logger << "subvolume shape " << sub_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";

	int epoch(0);
	//MCMC
	int rejects(0);

	cout << "Starting MH" << endl;
	if (false) {
	// if (argc == 1) {
		copy_subvolume(tiff_data, net.input(), tiff_label, label_subset);
		net.input().draw_slice("img/mh_sub_input.png",0);
		label_subset.draw_slice("img/mh_sub_label.png",0);

		net.e += .01;

		while (true) {
			static float SIGMA = 0.005;
			static float SIGMA_DECAY = pow(.5, 1.0 / 50);
			static float last_loss = 9999999;


			net.a.init_normal(0, 1.0);
			net.b = net.param;
			net.c = net.a;
			net.c *= net.e;

			net.param += net.c;


			net.forward();
			ostringstream oss;
			oss << "img/mh-" << epoch << ".png";
			net.output().draw_slice(oss.str(), 3);
			//cout << net.output().to_vector() << endl;
			//cout << net.param.to_vector() << endl;
			float loss = net.calculate_loss(label_subset) / sub_shape.size();
			logger << "epoch: " << epoch << ": loss " << loss << "\n";

			if (loss < last_loss || (exp(-loss/SIGMA) / exp(-last_loss/SIGMA)) > rand_float()) {
				last_loss = loss;
				cout << "accept" << endl;
				rejects = 0;
				net.a.pow(2.0);
				net.a += -1.;
				net.a *= .1; //LR
				net.a.exp();
				net.e *= net.a;
				// cout << "new vec: " << net.e.to_vector() << endl;

				net.save(netname);
			}
			else{
				net.param = net.b;
				cout << "reject" << endl;
				++rejects;
			}
			if (rejects > 10)
				break;
			SIGMA *= SIGMA_DECAY;
			cout << "sigma: " << SIGMA << endl;
			++epoch;
		}
	}

	cout << "Starting MOM-RMSPROP" <<endl;
	epoch = 0;
	float last_loss = 9999999.;

	float base_lr = .01;
	float const LR_DECAY = pow(.5, 1.0 / 200);

	while (true) {
		ostringstream ose;
		ose << "img/mom_sub_in-" << epoch << ".png";
		copy_subvolume(tiff_data, net.input(), tiff_label, label_subset, rand()%2, rand()%2, rand()%2, rand()%2);

		//copy_subvolume(tiff_data, net.input(), tiff_label, label_subset, true);
		net.input().rand_zero(.2);
		net.input().draw_slice(ose.str(),0);

		// return 1;
		ostringstream osse;
		osse << "img/mom_sub_label-" << epoch << ".png";
		label_subset.draw_slice(osse.str(),0);

		Timer ftimer;
		net.forward();
		cout << "forward took:" << ftimer.since() << endl;

		ostringstream oss;
		oss << "img/mom-rmsprop-" << epoch << ".png";
		net.output().draw_slice(oss.str(), 0);
		//cout << net.output().to_vector() << endl;
		//cout << net.param.to_vector() << endl;
		float loss = net.calculate_loss(label_subset);
		logger << "epoch: " << epoch << ": loss " << (loss / sub_shape.size()) << "\n";
		if (loss < last_loss) {
			last_lo ss = loss;
			net.save(netname);

		}

		Timer timer;
		net.backward();
		cout << "backward took:" << timer.since() << "\n\n";
		net.grad *= 1.0 / tiff_data.size();

		// cout << net.grad.to_vector() << endl;
		// net.update(.1);

		//SGD
		// net.grad *= .00001;
		// net.param += net.grad;

		//RMS PROP
		float decay = epoch < 4 ? 0.5 : 0.9;
		float mean_decay = epoch < 4 ? 0.5 : 0.9;
		float eps = .00001;
		//float lr = 0.001;
		//float lr = 0.01;

		// float lr = epoch < 4 ? .0001 : .001;

		float lr = .001 + base_lr;
		base_lr *= LR_DECAY;
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

		//Marijn Trick

		//net.d = net.c;
		//net.d *= (1.0 - mean_decay);
		//net.e *= mean_decay;
		//net.e += net.d;

		//net.d = net.e;
		//net.d.abs();
		//net.c *= net.d;

		//Momentum

		net.d = net.c;
		net.d *= (1.0 - mean_decay);
		net.e *= mean_decay;
		net.e += net.d;
		net.c = net.e;

		//update
		//net.c.clip(1.);
		net.c *= lr;
		net.param += net.c;

		print_last(net.grad.to_vector(), 10);
		print_last(net.rmse.to_vector(), 10);
		print_last(net.e.to_vector(), 10);
		print_last(net.param.to_vector(), 10);


		++epoch;
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
