#include <iostream>
#include <cuda.h>
#include <sstream>

#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
//#include "tiff.h"
#include "log.h"
#include "divide.h"
#include "raw.h"
#include "trainer.h"

using namespace std;




int main(int argc, char **argv) {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);


	string img_path = "img-brain/";

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
	// // Volume outputs[brainnum] = open_tiff("7nm/input.tif", true);
	// Volume tiff_label = open_tiff("7nm/binary-labels.tif", false, true);

	int n_brains(5);
	int width = 240;
	int height = 240;
	int depth = 48;
	vector<Volume> inputs(n_brains);
	vector<Volume> outputs(n_brains);

	for (size_t n(0); n < n_brains; ++n) {
		ostringstream oss1, oss2, oss3, oss4, oss5,oss_label;
		oss1 << "../mrbrain-raw/TrainingData/" << n+1 <<"/"<< "T1.raw";
		oss2 << "../mrbrain-raw/TrainingData/" << n+1 <<"/"<< "T1_IR_PP.raw";
		oss3 << "../mrbrain-raw/TrainingData/" << n+1 <<"/"<< "T2_FLAIR.raw";
		oss4 << "../mrbrain-raw/TrainingData/" << n+1 <<"/"<< "T1_PP.raw";
		oss5 << "../mrbrain-raw/TrainingData/" << n+1 <<"/"<< "T2_FLAIR_PP.raw";
		inputs[n] = open_raw5(oss1.str(), oss2.str(), oss3.str(), oss4.str(), oss5.str(), width, height, depth);
		cout << inputs[n].shape << " " << inputs[n].buf->n << endl;

		inputs[n].draw_slice_rgb("input.bmp",10);

		//inputs[n].draw_slice_rgb("temp.bmp",0);

		oss_label << "../mrbrain-raw/TrainingData/" << n+1 <<"/"<< "LabelsForTraining.raw";
		outputs[n] = open_raw(oss_label.str(), width, height, depth);
		// outputs[n].draw_slice_rgb("label.bmp",10);

	}

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
	VolumeShape sub_shape = inputs[0].shape;
	// Volume in_data(shape);
	// Volume out_data(shape);

	// in_data.init_normal(0, .5);
	// out_data.init_normal(0, .5);
	sub_shape.w = 240;
	sub_shape.h = 240;
	sub_shape.z = 25;
	// sub_shape.w = 32; sub_shape.h = 32; sub_shape.z = 16; sub_shape.w =
	// 256; sub_shape.h = 256; sub_shape.z = 8;

	//We need a volume for sub targets
	Volume label_subset(VolumeShape{sub_shape.z, outputs[0].shape.c, sub_shape.w, sub_shape.h});

	VolumeNetwork net(sub_shape);

	// net.add_fc(8);
	// net.add_vlstm(7, 7, 8);
	// net.add_fc(8);
	// net.add_tanh();
	// net.add_fc(2);
	// net.add_softmax();


	//Marijn net
	//Wonmin net
	// net.add_fc(16);
	// net.add_vlstm(7, 7, 16);
	// net.add_fc(32);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 32);
	// net.add_fc(64);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 64);
	// net.add_fc(64);
	// net.add_tanh();
	// net.add_fc(5);
	// net.add_softmax();

	//Wonmin net
	//// net.add_fc(16);
	net.add_vlstm(7, 7, 16);
	net.add_fc(32);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(32);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(128);
	net.add_tanh();
	net.add_fc(5);
	net.add_softmax();

	//Wonmin net2
	/*net.add_fc(32);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(64, .1);
	net.add_tanh();
	net.add_vlstm(7, 7, 64);
	net.add_fc(128, .1);
	net.add_tanh();
	net.add_vlstm(7, 7, 128);
	net.add_fc(256, .1);
	net.add_tanh();
	net.add_fc(2);
	net.add_softmax();
	*/


	net.finish();
	// net.init_normal(0, .1);
	// net.init_uniform(.1);

	cout << net.volumes[0]->x.shape << endl;
	cout << inputs[0].shape << endl;
	// net.set_input(tiff_data);
	// net.volumes[0]->x.draw_slice("in_3.png", 3);
	// tiff_label.draw_slice("label_3.png", 3);

	if (argc > 1) {
	  net.load(argv[1]);
	}
	else{
		net.init_uniform(.1);
	}

	logger << "begin description\n";
	logger << "subvolume shape " << sub_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";

	int epoch(0);

	cout << "Starting MOM-RMSPROP" <<endl;
	epoch = 0;
	float last_loss = 9999999.;

	int n_sums(50); // marijn trick vars
	int sum_counter(0);
	int burnin(50);

	// Trainer trainer(net.param_vec.n, .01, .0000001, 200);
	Trainer trainer(net.param_vec.n, .03, .0001, 1000);

	while (true) {
	  Timer total_timer;
		ostringstream ose;
		ose << img_path << "mom_sub_in-" << epoch << ".png";
		int brainnum = rand() % n_brains;
		copy_subvolume(inputs[brainnum], net.input(), outputs[brainnum], label_subset, false, rand()%2, false, false); // rotate, xflip, yflip, zflip

		//copy_subvolume(inputs[brainnum], net.input(), outputs[brainnum], label_subset, true);
		//
		//net.input().rand_zero(.2);
		net.input().draw_slice_rgb(ose.str(),0);

		// return 1;
		ostringstream osse;
		osse << img_path << "mom_sub_label-" << epoch << ".png";
		label_subset.draw_slice_rgb(osse.str(),0);

		Timer ftimer;
		cout << "------------" << endl;
		// print_last(net.param_vec.to_vector(), 30);

		net.forward();
		// print_last(net.param_vec.to_vector(), 30);

		cout << "forward took:" << ftimer.since() << endl;

		ostringstream oss;
		oss << img_path << "mom-rmsprop-" << epoch << ".png";
		net.output().draw_slice_rgb(oss.str(), 0);
		//cout << net.output().to_vector() << endl;
		//cout << net.param.to_vector() << endl;
		float loss = net.calculate_loss(label_subset);
		logger << "epoch: " << epoch << ": loss " << (loss / sub_shape.size()) << "\n";
		//if (loss < last_loss) {
		last_loss = loss;
		net.save(netname);

		//}

		Timer timer;
		// cout << "=====" << endl;
		// print_last(net.param_vec.to_vector(), 30);

		net.backward();
		// print_last(net.param_vec.to_vector(), 30);

		cout << "backward took:" << timer.since() << "\n\n";
		net.grad_vec *= 1.0 / inputs[brainnum].size();

		// cout << "before after" << endl;
		// print_last(net.param_vec.to_vector(), 30);
		trainer.update(&net.param_vec, net.grad_vec);
		// cout << "-------" << endl;
		// print_last(net.param_vec.to_vector(), 30);
		// cout << "=====end" << endl;
		// float lr = epoch < 4 ? .0001 : .001;

		cout << "grad: ";
		print_wide(net.grad_vec.to_vector(), 20);
		// print_last(net.rmse.to_vector(), 10);
		// print_last(net.e.to_vector(), 10);
		cout << "param: ";
		print_wide(net.param_vec.to_vector(), 20);


		++epoch;
		cout << "epoch time: " << total_timer.since() <<" lr: " << trainer.lr() << endl;
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
	// 	//cout << outputs[brainnum].to_vector() << endl;
	// 	// vlstm.operations[4]->volumes["c"]->x.draw_slice("c.png", 4);
	// 	vlstm.y.x.draw_slice("out.png", 8);
	// 	vlstm.x.x.draw_slice("in.png", 8);
	// 	outputs[brainnum].draw_slice("target.png", 8);

	// 	vlstm.y.diff.from_volume(outputs[brainnum]);
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
