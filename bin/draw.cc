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

//normalise to U(0, 1)
inline void normalise(vector<float> *values_p) {
	auto &values(*values_p);
	float mean(0), std(0);
	for (auto v : values) {
		mean += v;
	}
	mean /= values.size();

	for (auto v : values) {
		std += (v - mean) * (v - mean);
	}
	std = sqrt(std / (values.size() - 1.0));

	for (auto &v : values)
		v = (v - mean) / std;
}

int main(int argc, char **argv) {
  // srand(time(0));
	srand(1342342);
	Handler::set_device(0);

	Log logger("log.txt");
	
	int img_w = 32;
	int img_h = 32;
	int img_c = 1;
	int train_n = 30;
	
	cout << "whc: " << img_w << " " << img_h << " " << img_c << endl;
	// cout << img.data()[10] << endl;



	//VolumeShape shape{100, 1, 512, 512};


	//int kg(3), ko(3), c(1);
	// VolumeShape train_shape{train_n, img_c + 44, img_w, img_h};
	VolumeShape train_shape{train_n, img_c, img_w, img_h};
	VolumeShape target_shape{train_n, img_c, img_w, img_h};

	int kg(7), ko(7), c(1);


	VolumeNetwork net(train_shape);

	net.add_fc(32);
	net.add_tanh();
	net.add_univlstm(7, 7, 16);
	net.add_univlstm(7, 7, 32);
	// net.add_univlstm(7, 7, 32);
	net.add_fc(64);
	net.add_tanh();
	// net.add_vlstm(7, 7, 16);
	// net.add_vlstm(7, 7, img_c);
	net.add_fc(img_c);

	// net.add_tanh();


	// net.add_tanh();
	// net.add_softmax();

	net.finish();
	// net.init_normal(0, .1);
	// net.init_uniform(.1);
	cout << net.param_vec.n << endl;

	logger << "begin net description\n";
	logger << "input volume shape " << train_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";


	//Fast-weight network
	int n_action(32);
	TensorShape action_input{train_n, n_action, 1, 1};
	// TensorShape action_input{train_n, 3, 1, 1};

	Network<float> fastweight_net(action_input);
	// fastweight_net.add_conv(16, 1, 1);
	// fastweight_net.add_tanh();
	// fastweight_net.add_conv(32, 1, 1);
	// fastweight_net.add_tanh();
	fastweight_net.add_conv(64, 1, 1);
	fastweight_net.add_tanh();
	fastweight_net.add_conv(32, 1, 1);
	fastweight_net.add_tanh();
	fastweight_net.add_conv(16, 1, 1);
	fastweight_net.add_tanh();

	fastweight_net.add_conv(net.fast_param_vec.n / train_n, 1, 1);

	fastweight_net.add_tanh();
	fastweight_net.finish();

	// fastweight_net.init_uniform(.1);
	// fastweight_net.init_normal(.0, .1);

	logger << "begin fastweight description\n";
	logger << "input volume shape " << train_shape << "\n";
	fastweight_net.describe(logger.file);
	logger << "end description\n";

	Volume input(train_shape), target(target_shape);

	int n_given(10);
	// int n_epoch = 200;
	// vector<float> rmse(train_n);
	vector<vector<float>> errors(train_n);
	
	Tensor<float> err_tensor(net.output().slice_shape()), target_tensor(net.output().slice_shape());

	int n_test(10);
	string exp_dir = "exp/";
	// srand(1342342);
	for (int index(0); index < n_test; ++index) {
		ostringstream epoch_path;
		epoch_path << exp_dir << index << "-";
		fastweight_net.input().init_normal(.0, .1);

		for (size_t i=n_given-1; i < train_n; ++i) {
			Timer fasttimer;
			fastweight_net.forward();

			cout << "fast forward took:" << fasttimer.since() << endl;

			net.set_fast_weights(fastweight_net.output());
			net.forward();

			if ((i + 1) < train_n)
				copy_gpu_to_gpu(net.output().slice(i), net.input().slice(i+1), net.output().slice_size);
		}

		// net.input().draw_slice(epoch_path.str() + "input_last.png",	train_n-1);
		// net.input().draw_slice(epoch_path.str() + "input_middle.png",	train_n / 2);
		// net.output().draw_slice(epoch_path.str() + "output_middle.png",train_n / 2);
		// net.output().draw_slice(epoch_path.str() + "output_last.png",train_n - 1);
		// cout << "output/target:" << endl;
		// print_wide(net.output().to_vector(), 30);
		// print_wide(target.to_vector(), 30);
		// target.draw_slice(epoch_path.str() + "target_middle.png",train_n/2);
		// target.draw_slice(epoch_path.str() + "target_last.png",train_n-1);


		// float loss = net.calculate_loss(target);
		// logger << "epoch: " << epoch << ": loss " << sqrt(loss / train_shape.size()) << "\n";
		// last_loss = loss;

	}

	int i(0);
	for (auto err_vec : errors) {
		float mean(0), var(0);
		for (auto v : err_vec)
			mean += v;
		mean /= err_vec.size();
		for (auto v : err_vec)
			var += (v - mean) * (v - mean);
		var = sqrt(var / (err_vec.size() - 1));
		cout << i << " " << mean << " " << var << " " << (var / sqrt(err_vec.size())) << endl;
		++i;
	}

	cudaDeviceSynchronize();
}
