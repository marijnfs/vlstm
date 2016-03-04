#include <iostream>
#include <cuda.h>
#include <sstream>

#include "network.h"
#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
#include "tiff.h"
#include "log.h"
#include "divide.h"
#include "database.h"
#include "img.pb.h"
#include "trainer.h"

using namespace std;

inline void random_next_step_subvolume(Database &db, Volume &input, Volume &target, Tensor<float> &actions) {
	int N = db.count("img");
	int diff = 1;
	int n = input.shape.z;
	int sub_n = n + diff; //1step prediction
	int start = rand() % (N - sub_n);
	vector<float> x_last;

	for (size_t i(0); i < sub_n; ++i) {
		// cout << i << endl;
		Img img = db.load<Img>("img", i + start);
		assert(img.w() == input.shape.w);
		assert(img.h() == input.shape.h);
		assert(img.c() == input.shape.c);

		Action action = db.load<Action>("action", i + start);
		vector<float> x(action.x().data(), action.x().data()+action.x().size());
		// cout << "copy:" << endl;
		// cout << img.data().data() << " " << input.slice(i) << " " << img.data().size() << endl;
		// vector<float> img_correct(img.data().size());
		// float const *it = img.data().data();
		// for (size_t c(0); c < img.c(); ++c)
		// 	for (size_t y(0); y < img.h(); ++y)
		// 		for (size_t x(0); x < img.w(); ++x, ++it)
		// 			img_correct[c * img.w() * img.h() + x * img.h() + y] = *it;
		// cout << "input shape " << input.shape << endl;
		if (i < n)
			copy_cpu_to_gpu<>(img.data().data(), input.slice(i),  img.data().size());
		if (i >= diff)
			copy_cpu_to_gpu<>(img.data().data(), target.slice(i-diff), img.data().size());
		if (i >= diff) {
			vector<float> a(x);
			for (size_t i(0); i < a.size(); ++i) a[i] -= x_last[i];
			// cout << a << endl;
			copy_cpu_to_gpu<>(&a[0], actions.ptr() + (i-diff) * a.size(), a.size());
		}
		x_last = x;

	}
	// throw "";
}

int main(int argc, char **argv) {
	Log logger("log.txt");
	Handler::set_device(0);

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
	int train_n = 20;


	cout << "whc: " << img_w << " " << img_h << " " << img_c << endl;
	// cout << img.data()[10] << endl;



	//VolumeShape shape{100, 1, 512, 512};


	//int kg(3), ko(3), c(1);
	VolumeShape train_shape{train_n, img_c, img_w, img_h};

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
	// net.add_fc(8);
	net.add_univlstm(7, 7, img_c);
	// net.add_fc(8);
	// net.add_tanh();

	// net.add_univlstm(7, 7, 4);
	//net.add_fc(32);
	//net.add_tanh();
	//net.add_vlstm(7, 7, 32);
	// net.add_fc(32);
	// net.add_tanh();
	// net.add_fc(img_c);
	// net.add_softmax();

	net.finish();
	// net.init_normal(0, .1);
	net.init_uniform(.1);


	if (argc > 1) {
	  net.load(argv[1]);
	}

	logger << "begin net description\n";
	logger << "input volume shape " << train_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";


	//Fast-weight network
	TensorShape action_input{train_n, 3, 1, 1};
	Tensor<float> actions_tensor(action_input);

	Network<float> fastweight_net(action_input);
	fastweight_net.add_conv(16, 1, 1);
	fastweight_net.add_tanh();
	fastweight_net.add_conv(32, 1, 1);
	fastweight_net.add_tanh();
	fastweight_net.add_conv(64, 1, 1);
	fastweight_net.add_tanh();
	fastweight_net.add_conv(net.fast_param_vec.n / train_n, 1, 1);
	// fastweight_net.add_tanh();
	fastweight_net.finish();
	fastweight_net.init_uniform(.1);

	logger << "begin fastweight description\n";
	logger << "input volume shape " << train_shape << "\n";
	fastweight_net.describe(logger.file);
	logger << "end description\n";

	int epoch(0);
	float last_loss = 9999999.;

	int n_sums(50); // marijn trick vars
	int sum_counter(0);
	int burnin(50);


	Volume input(train_shape), target(train_shape);

	Trainer trainer(net.param_vec.n, .01, .00001, 40);
	Trainer fast_trainer(fastweight_net.n_params, .01, .00001, 40);


	while (true) {
		random_next_step_subvolume(db, net.input(), target, fastweight_net.input());

		Timer fasttimer;
		fastweight_net.forward();
		cout << "fast forward took:" << fasttimer.since() << endl;
		// cout << fastweight_net.output().to_vector() << endl;
		// cout << fastweight_net.input().to_vector() << endl;

		net.set_fast_weights(fastweight_net.output());
		//cout << net.fast_param_vec.to_vector() << endl;
		// cout << net.param_vec.to_vector() << endl;

	    Timer total_timer;
		net.input().draw_slice("input_last.png",	train_n-1);

		Timer ftimer;
		net.forward();
		net.output().draw_slice("output_last.png",train_n-1);
		target.draw_slice("target_last.png",train_n-1);

		cout << "forward took:" << ftimer.since() << endl;

		float loss = net.calculate_loss(target);
		logger << "epoch: " << epoch << ": loss " << (loss / train_shape.size()) << "\n";
		last_loss = loss;
		//net.save(netname);

		Timer timer;
		// cout << last(net.volumes)->diff.to_vector() << endl;
		net.backward();
		cout << "backward took:" << timer.since() << "\n\n";

		// cout << net.fast_grad_vec.to_vector() << endl;
		// return 0;
		net.get_fast_grads(fastweight_net.output_grad());
		// cout << fastweight_net.output_grad().to_vector() << endl;
		trainer.update(&net.param_vec, net.grad_vec);

		// return 0;
		fastweight_net.backward();
		fast_trainer.update(&fastweight_net.param_vec, fastweight_net.grad_vec);

		++epoch;
		cout << "epoch time: " << total_timer.since() << endl;
		// return 0;
	}

	cudaDeviceSynchronize();
}
