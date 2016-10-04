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

// #include "img.h"
inline void next_step_subvolume(Database &db, Volume &input, Volume &target, Tensor<float> &actions, int index) {
	int N = db.count("img");
	int Noffset = N * 9 / 10;//Training set
	N = N - Noffset;//Training set
	int diff = 1;
	int n = input.shape.z;
	int sub_n = n + diff; //1step prediction
	int start = index + Noffset;
	vector<float> x_last, q_last;

	for (size_t i(0); i < sub_n; ++i) {
		// cout << i << endl;
		Img img = db.load<Img>("img", i + start);
		// write_img1c("test.png", img.w(), img.h(), img.data().data());
		// throw StringException("stop");
		// assert(img.w() == input.shape.w);
		// assert(img.h() == input.shape.h);
		// assert(img.c() == input.shape.c);

		Action action = db.load<Action>("action", i + start);
		vector<float> x(action.x().data(), action.x().data()+action.x().size());
		vector<float> q(action.q().data(), action.q().data()+action.q().size());
		// cout << "copy:" << endl;
		// cout << img.data().data() << " " << input.slice(i) << " " << img.data().size() << endl;
		vector<float> img_correct(input.shape.w * input.shape.h * input.shape.c);
		// float const *it = img.data().data();
		// for (size_t c(0); c < img.c(); ++c)

		//one channel only, scale down by half
		for (size_t y(0); y < input.shape.h; ++y)
			for (size_t x(0); x < input.shape.w; ++x)
				for (size_t dx(0); dx < 8; ++dx)
					for (size_t dy(0); dy < 8; ++dy)
			 			img_correct[y * input.shape.w + x] = img.data((y * 8 + dy) * img.w() + x * 8 + dx) / (8. * 8.);

		normalise(&img_correct);
		// cout << "input shape " << input.shape << endl;
		if (i < n)
			copy_cpu_to_gpu<>(&img_correct[0], input.slice(i),  img_correct.size());

		// if (i < n)
			// copy_cpu_to_gpu<>(&img_correct[0], target.slice(i), img_correct.size());
		if (i >= diff)
			copy_cpu_to_gpu<>(&img_correct[0], target.slice(i-diff), img_correct.size());

		if (i >= diff) {
			vector<float> a(x);
			for (size_t i(0); i < a.size(); ++i) a[i] -= x_last[i];
			vector<float> qa(q);
			for (size_t i(0); i < qa.size(); ++i) qa[i] -= q_last[i];
			copy(qa.begin(), qa.end(), back_inserter(a));
			assert(a.size() == actions.shape.dcs);
			// cout << a << endl;
			// copy_cpu_to_gpu<>(&a[0], actions.ptr() + (i-diff) * a.size(), a.size());
			copy_cpu_to_gpu<>(&a[0], actions.ptr() + (i-diff) * a.size(), a.size());
		}
		x_last = x;
		q_last = q;

	}
	// throw "";
}

inline void next_step_subvolume_added_info(Database &db, Volume &input, Volume &target, Tensor<float> &actions, int index) {
	int N = db.count("img");
	N = N * 9 / 10;//Training set
	int diff = 1;
	int n = input.shape.z;
	int sub_n = n + diff; //1step prediction
	int start = N + index;
	vector<float> x_last, q_last;

	int n_channel = 1;
	int n_action = 44;
	if (input.shape.c != n_channel + n_action)
		throw StringException("not right amount of input volume channels");

	for (size_t i(0); i < sub_n; ++i) {
		// cout << i << endl;
		Img img = db.load<Img>("img", i + start);
		// assert(img.w() == input.shape.w);
		// assert(img.h() == input.shape.h);
		// assert(img.c() == input.shape.c);
		vector<float> img_correct(input.shape.w * input.shape.h * n_channel);
		//one channel only, scale down by half
		for (size_t y(0); y < input.shape.h; ++y)
			for (size_t x(0); x < input.shape.w; ++x)
				for (size_t dx(0); dx < 8; ++dx)
					for (size_t dy(0); dy < 8; ++dy)
			 			img_correct[y * input.shape.w + x] = img.data((y * 8 + dy) * img.w() + x * 8 + dx) / (8. * 8.);

		normalise(&img_correct);

		Action action = db.load<Action>("action", i + start);
		vector<float> x(action.x().data(), action.x().data()+action.x().size());
		vector<float> q(action.q().data(), action.q().data()+action.q().size());
		// cout << "copy:" << endl;
		// cout << img.data().data() << " " << input.slice(i) << " " << img.data().size() << endl;
		// float const *it = img.data().data();
		// for (size_t c(0); c < img.c(); ++c)

		// cout << "input shape " << input.shape << endl;
		if (i < n)
			copy_cpu_to_gpu<>(&img_correct[0], input.slice(i),  img_correct.size());

		// if (i < n)
			// copy_cpu_to_gpu<>(&img_correct[0], target.slice(i), img_correct.size());
		if (i >= diff)
			copy_cpu_to_gpu<>(&img_correct[0], target.slice(i-diff), img_correct.size());

		if (i >= diff) {
			vector<float> img_action(input.shape.w * input.shape.h * n_action);

			vector<float> a(x);
			for (size_t i(0); i < a.size(); ++i) a[i] -= x_last[i];

			vector<float> qa(q);
			for (size_t i(0); i < qa.size(); ++i) qa[i] -= q_last[i];

			copy(qa.begin(), qa.end(), back_inserter(a)); //a contains full action vec
			assert(a.size() == actions.shape.dcs);
			// cout << a << endl;
			// copy_cpu_to_gpu<>(&a[0], actions.ptr() + (i-diff) * a.size(), a.size());
			copy_cpu_to_gpu<>(&a[0], actions.ptr() + (i-diff) * a.size(), a.size());
			cout << input.shape << endl;
			cout << img_action.size() << endl;
			cout << img_correct.size() << endl;
			for (size_t n(0); n < a.size(); ++n)
				fill(&img_action[n * input.shape.w * input.shape.h], &img_action[(n+1) * input.shape.w * input.shape.h], a[n]);
			cout << i << " " << diff << endl;
			copy_cpu_to_gpu<>(&img_action[0], input.slice(i-diff) + img_correct.size(), img_action.size()); //add action images to end of regular image

		}
		x_last = x;
		q_last = q;

	}
	// throw "";
}

int main(int argc, char **argv) {
	// srand(time(0));
	srand(1342342);
	Handler::set_device(0);

	string exp_dir("exp-uni-eval/");
	Log logger(exp_dir + "log.txt");

	Database db("/home/cvlstm/data/exp-march7-30min.db");
	cout << db.count("exp") << endl;
	cout << db.count("img") << endl;
	cout << db.count("action") << endl;

	Experiment exp = db.load<Experiment>("exp", 0);
	Img img = db.load<Img>("img", 0);
	vector<float> data(img.data().size());
	copy(img.data().begin(), img.data().end(), data.begin());

	// int img_w = img.w();
	// int img_h = img.h();
	// int img_c = img.c();

	int img_w = img.w()/8;
	int img_h = img.h()/8;
	int img_c = 1;
	int train_n = 30;
	int n_test = db.count("img")/10 - train_n;

	cout << "whc: " << img_w << " " << img_h << " " << img_c << endl;
	// cout << img.data()[10] << endl;



	//VolumeShape shape{100, 1, 512, 512};


	//int kg(3), ko(3), c(1);
	// VolumeShape train_shape{train_n, img_c + 44, img_w, img_h};
	VolumeShape train_shape{train_n, img_c, img_w, img_h};
	VolumeShape target_shape{train_n, img_c, img_w, img_h};

	int kg(7), ko(7), c(1);


	VolumeNetwork net(train_shape);

	// net.add_fc(8);
	// net.add_vlstm(7, 7, 8);
	// net.add_fc(8);
	// net.add_tanh();
	// net.add_fc(2);
	// net.add_softmax();


	//Marijn net
	// net.add_fc(8);
	// net.add_univlstm(7, 7, 8);
	// net.add_fc(8);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 8);
	// net.add_univlstm(7, 7, 2);
	// net.add_univlstm(7, 7, 4);
	// net.add_univlstm(7, 7, img_c);
	// net.add_univlstm(11, 11, 2);
	// net.add_univlstm(11, 11, 32);
	// net.add_univlstm(5, 5, 128);
	// net.add_univlstm(9, 9, 16);
	// net.add_univlstm(9, 9, 16);
	// net.add_univlstm(7, 7, img_c);
	// net.add_univlstm(7, 7, 32);
	// net.add_univlstm(7, 7, 16);
	// net.add_univlstm(7, 7, img_c);
	// net.add_univlstm(7, 7, 64);
	// net.add_fc(8);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 32);
	// net.add_fc(16);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 16);
	// net.add_vlstm(7, 7, 32);
	// net.add_vlstm(7, 7, 64);


	// net.add_fc(img_c);
	// net.add_tanh();

	// net.add_fc(img_c);
	// net.add_tanh();

	// net.add_fc(32);
	// net.add_tanh();
	// net.add_fc(32);
	// net.add_tanh();
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
	TensorShape action_input{train_n, 3+41, 1, 1};
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

	if (argc > 2) {
	  net.load(argv[1]);
	  fastweight_net.load(argv[2]);

	} else {
		throw StringException("need to provide net file");
	}

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

	// srand(1342342);
	for (int index(0); index < n_test; ++index) {
		ostringstream epoch_path;
		epoch_path << exp_dir << index << "-";
		next_step_subvolume(db, net.input(), target, fastweight_net.input(), index);
		// next_step_subvolume_added_info(db, net.input(), target, fastweight_net.input(), index);
		// net.input().zero(n_given); //zero out from image 10
		// net.input().draw_slice("test1.png", 9);
		// net.input().draw_slice("test2.png", 10);
		for (size_t n(n_given); n < net.input().shape.z; ++n)
			cudaMemset(net.input().buf->data + n * net.input().slice_size, 0, sizeof(float) * net.output().slice_size);
		// net.input().draw_slice("test3.png", 9);
		// net.input().draw_slice("test4.png", 10);


		// handle_error( cudaMemset(data + offset, 0, sizeof(float) * (n - offset) ) );

		// if (epoch % 100 == 0)
		// cout << "fastweight input: " << fastweight_net.input().shape() << " " << fastweight_net.input().to_vector() << endl;

	    Timer total_timer;


		for (size_t i=n_given-1; i < train_n; ++i) {
			Timer fasttimer;
			fastweight_net.forward();

			cout << "fast forward took:" << fasttimer.since() << endl;

			net.set_fast_weights(fastweight_net.output());

			Timer ftimer;
			net.forward();
			cout << "forward took:" << ftimer.since() << endl;

			if (index % 20 == 0) {
				ostringstream oss;
				oss << exp_dir << "t-" << index << "-prediction-" << i << ".png";
				ostringstream oss2;
				oss2 << exp_dir << "t-" << index << "-target-" << i << ".png";
				net.output().draw_slice(oss.str(), i);
				target.draw_slice(oss2.str(), i);
			}

			err_tensor.from_ptr(net.output().slice(i));
			target_tensor.from_ptr(target.slice(i));
			err_tensor -= target_tensor;
			// rmse[i] += err_tensor.norm2() / err_tensor.size();
			cout << "ERR:" << err_tensor.norm2() << " " << err_tensor.size() << endl;

			errors[i].push_back(err_tensor.norm2() / err_tensor.size());
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
