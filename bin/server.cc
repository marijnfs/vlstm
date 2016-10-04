#include <iostream>
#include <cuda.h>
#include <sstream>
// mutex::lock/unlock
#include <thread>         // std::thread
#include <mutex>          // std::mutex

std::mutex mtx;           // mutex for critical section

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


	int kg(7), ko(7), c(1);

	Volume label_subset(VolumeShape{sub_shape.z, tiff_label.shape.c, sub_shape.w, sub_shape.h});
	VolumeNetwork net(sub_shape);


	//Marijn net
	net.add_fc(8);
	net.add_vlstm(7, 7, 32);
	//net.add_fc(32);
	//net.add_tanh();
	net.add_vlstm(7, 7, 32);
	//net.add_fc(32);
	//net.add_tanh();
	//net.add_vlstm(7, 7, 32);
	net.add_fc(32);
	net.add_tanh();
	net.add_fc(2);
	net.add_softmax();

	net.finish();
	net.init_uniform(.1);

	//if (argc > 1) {
	// net.load(argv[1]);
	//}

	logger << "begin description\n";
	logger << "subvolume shape " << sub_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";

	int epoch(0);

	cout << "Starting MOM-RMSPROP" <<endl;
	epoch = 0;
	float last_loss = 9999999.;

	float base_lr = .01;
	float const LR_DECAY = pow(.5, 1.0 / 100);

	int n_sums(50); // marijn trick vars
	int sum_counter(0);
	int burnin(50);

	while (true) {
	  Timer total_timer;
		ostringstream ose;
		ose << img_path << "mom_sub_in-" << epoch << ".png";
		copy_subvolume(tiff_data, net.input(), tiff_label, label_subset, rand()%2, rand()%2, rand()%2, rand()%2);

		//copy_subvolume(tiff_data, net.input(), tiff_label, label_subset, true);
		//
		//net.input().rand_zero(.2);
		net.input().draw_slice(ose.str(),0);

		// return 1;
		ostringstream osse;
		osse << img_path << "mom_sub_label-" << epoch << ".png";
		label_subset.draw_slice(osse.str(),0);

		Timer ftimer;
		net.forward();
		cout << "forward took:" << ftimer.since() << endl;

		ostringstream oss;
		oss << img_path << "mom-rmsprop-" << epoch << ".png";
		net.output().draw_slice(oss.str(), 0);
		//cout << net.output().to_vector() << endl;
		//cout << net.param.to_vector() << endl;
		float loss = net.calculate_loss(label_subset);
		logger << "epoch: " << epoch << ": loss " << (loss / sub_shape.size()) << "\n";
		//if (loss < last_loss) {
		last_loss = loss;
		net.save(netname);

		//}

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
		float mean_decay = 0.9;
		float eps = .00001;
		//float lr = 0.001;
		//float lr = 0.01;

		// float lr = epoch < 4 ? .0001 : .001;

		float lr = .00001 + base_lr;
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

		//Marijn Trick 2

		// if (epoch >= burnin) {
		//   net.d = net.param;
		//   net.d *= (1.0 / n_sums);
		//   net.e += net.d;
		//   ++sum_counter;

		//   if (sum_counter == n_sums) {
		//     net.param = net.e;
		//     net.e.zero();
		//     net.c.zero();
		//     sum_counter = 0;
		//     net.save("mean.net");
		//   }

		// }

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
		cout << "epoch time: " << total_timer.since() << endl;
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
