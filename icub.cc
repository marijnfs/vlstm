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
	for (size_t i(0); i < sub_n; ++i) {
		cout << i << endl;
		Img img = db.load<Img>("img", i + start);
		assert(img.w == input.shape.w);
		assert(img.h == input.shape.h);
		assert(img.c == input.shape.c);

		Action action = db.load<Action>("action", i + start);
		cout << "copy:" << endl;
		cout << img.data().data() << " " << input.slice(i) << " " << img.data().size() << endl;
		cout << "input shape " << input.shape << endl;
		if (i < n)
			copy_to_gpu<>(img.data().data(), input.slice(i),  img.data().size());
		if (i >= diff)
			copy_to_gpu<>(img.data().data(), target.slice(i-diff), img.data().size());
		if (i < n) {
			copy_to_gpu<>(action.a().data(), actions.ptr() + i * action.a().size(), action.a().size());
		}

	}
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
	net.add_fc(8);
	net.add_univlstm(7, 7, 4);
	net.add_fc(8);
	net.add_tanh();

	net.add_univlstm(7, 7, 4);
	//net.add_fc(32);
	//net.add_tanh();
	//net.add_vlstm(7, 7, 32);
	// net.add_fc(32);
	// net.add_tanh();
	net.add_fc(img_c);
	net.add_softmax();

	net.finish();
	//net.init_normal(0, .1);
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
	cout << "== " << net.fast_param.n << " " << train_n << endl;
	fastweight_net.add_conv(net.fast_param.n / train_n, 1, 1);
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

	Trainer trainer(net.param.n, .01, .00001, 40);
	Trainer fast_trainer(fastweight_net.n_params, .01, .00001, 40);


	while (true) {
		random_next_step_subvolume(db, net.input(), target, fastweight_net.input());

		Timer fasttimer;
		fastweight_net.forward();
		cout << "fast forward took:" << fasttimer.since() << endl;

		net.set_fast_weights(fastweight_net.output());

	    Timer total_timer;
		net.input().draw_slice("slice.png",0);

		Timer ftimer;
		net.forward();
		cout << "forward took:" << ftimer.since() << endl;

		cout << net.input().shape << " " << target.shape << endl;
		float loss = net.calculate_loss(target);
		logger << "epoch: " << epoch << ": loss " << (loss / train_shape.size()) << "\n";
		last_loss = loss;
		net.save(netname);

		Timer timer;
		net.backward();
		cout << "backward took:" << timer.since() << "\n\n";
		net.grad *= train_shape.size(); //loss is counter for every pixel, normalise

		trainer.update(&net.param, net.grad);

		net.get_fast_grads(fastweight_net.output_grad());
		net.backward();

		++epoch;
		cout << "epoch time: " << total_timer.since() << endl;
		return 0;
	}

	cudaDeviceSynchronize();
}
