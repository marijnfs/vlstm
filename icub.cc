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
#include "database.h"
#include "img.pb.h"

using namespace std;

void print_last(vector<float> vals, int n) {
	for (size_t i(vals.size() - n); i < vals.size(); ++i)
		cout << vals[i] << " ";
	cout << endl;
}

void random_next_step_subvolume(Database &db, Volume &input, Volume &target, int n) {
	int N = db.count("img");
	int diff = 1;
	int sub_n = n + diff; //1step prediction
	int start = rand() % (N - sub_n);
	for (size_t i(start); i < start + sub_n; ++i) {
		Img img = db.load<Img>("img", i);
		assert(img.w == input.shape.w);
		assert(img.h == input.shape.h);
		assert(img.c == input.shape.c);

		Action action = db.load<Action>("action", i);
		if (i < start + sub_n + diff)
			copy(img.data().begin(), img.data().end(), input.slice(i));
		if (i >= diff)
			copy(img.data().begin(), img.data().end(), target.slice(i-diff));
	}
}

int main(int argc, char **argv) {
	Database db("/home/cvlstm/data/exp3.db");
	cout << db.count("exp") << endl;
	cout << db.count("img") << endl;
	cout << db.count("action") << endl;

	Experiment exp = db.load<Experiment>("exp", 0);
	Img img = db.load<Img>("img", 0);
	vector<float> data(img.data().size());
	copy(img.data().begin(), img.data().end(), data.begin());

	int img_w = img.w();
	int img_h = img.h();
	int img_c = img.c();
	int train_n = 60;


	cout << "whc: " << img_w << " " << img_h << " " << img_c << endl;
	// cout << img.data()[10] << endl;

	VolumeShape train_shape{train_n, img_c, img_w, img_h};
	Volume input(train_shape), target(train_shape);

	Log logger("log.txt");
	return 0;

	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);


	//int kg(3), ko(3), c(1);
	int kg(7), ko(7), c(1);

	string netname = "net.save";
	VolumeNetwork net(train_shape);

	// net.add_fc(8);
	// net.add_vlstm(7, 7, 8);
	// net.add_fc(8);
	// net.add_tanh();
	// net.add_fc(2);
	// net.add_softmax();


	//Marijn net
	net.add_fc(8);
	net.add_vlstm(7, 7, 4);
	//net.add_fc(32);
	//net.add_tanh();
	net.add_vlstm(7, 7, 4);
	//net.add_fc(32);
	//net.add_tanh();
	//net.add_vlstm(7, 7, 32);
	net.add_fc(32);
	net.add_tanh();
	net.add_fc(2);
	net.add_softmax();

	net.finish();
	//net.init_normal(0, .1);
	net.init_uniform(.1);


	if (argc > 1) {
	  net.load(argv[1]);
	}

	logger << "begin description\n";
	logger << "input volume shape " << train_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";

	int epoch(0);
	float last_loss = 9999999.;
	float base_lr = .01;
	float const LR_DECAY = pow(.5, 1.0 / 100);

	int n_sums(50); // marijn trick vars
	int sum_counter(0);
	int burnin(50);

	string img_path = ".";

	while (true) {
	  Timer total_timer;
		net.input().draw_slice("slice.png",0);

		Timer ftimer;
		net.forward();
		cout << "forward took:" << ftimer.since() << endl;

		float loss = net.calculate_loss(target);
		logger << "epoch: " << epoch << ": loss " << (loss / train_shape.size()) << "\n";
		last_loss = loss;
		net.save(netname);

		Timer timer;
		net.backward();
		cout << "backward took:" << timer.since() << "\n\n";
		net.grad *= train_shape.size(); //loss is counter for every pixel, normalise

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

	cudaDeviceSynchronize();
}
