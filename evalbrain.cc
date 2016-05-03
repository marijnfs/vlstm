#include <iostream>
#include <cuda.h>
#include <sstream>

#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
#include "raw.h"
#include "tiff.h"
#include "log.h"
#include "divide.h"
#include "global.h"
#include <iomanip>

using namespace std;


int main(int argc, char **argv) {
  //Global::validation() = true;
	if (argc < 4) {
		cout << "usage: eval [net] [out] [start]" << endl;
		return 1;
	}


	istringstream issarg;
	issarg.str(argv[3]);
	int start(0);
	issarg >> start;

	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cout << prop.name << endl;

	int n_brains(15);
	int width = 240;
	int height = 240;
	int depth = 48;
	vector<Volume> inputs(n_brains);
	vector<Volume> outputs(n_brains);


	//sub_shape.w = 128;
	//sub_shape.h = 128;
	int nstep_x = 1;
	int nstep_y = 1;
	int nstep_z = 5;

	VolumeShape data_shape{48, 5, 240, 240};
	VolumeShape sub_shape = data_shape;
	sub_shape.z = 25;

	//We need a volume for sub targets
	VolumeNetwork net(sub_shape);


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

	// // //Wonmin net
	//net.add_fc(16);
	// net.add_vlstm(7, 7, 16);
	// net.add_fc(25);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 32);
	// net.add_fc(45);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 64);
	// // net.add_fc(32);
	// // net.add_tanh();
	// net.add_fc(5);
	// net.add_softmax();

	//Wonmin net2
	// net.add_fc(32);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 32);
	// net.add_fc(64, .5);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 64);
	// net.add_fc(128, .5);
	// net.add_tanh();
	// net.add_vlstm(7, 7, 128);
	// net.add_fc(256, .5);
	// net.add_tanh();
	// net.add_fc(2);
	// net.add_softmax();

	net.finish();
	//net.init_normal(0, .1);
	net.load(argv[1]);
	//net.init_uniform(.1);

	for (size_t n(start); n < n_brains; ++n) {
		ostringstream oss1, oss2, oss3, oss4, oss5;
		oss1 << "../mrbrain-raw/TestData/" << n+1 <<"/"<< "T1.raw";
		oss2 << "../mrbrain-raw/TestData/" << n+1 <<"/"<< "T1_IR_PP.raw";
		oss3 << "../mrbrain-raw/TestData/" << n+1 <<"/"<< "T2_FLAIR.raw";
		oss4 << "../mrbrain-raw/TestData/" << n+1 <<"/"<< "T1_PP.raw";
		oss5 << "../mrbrain-raw/TestData/" << n+1 <<"/"<< "T2_FLAIR_PP.raw";
		Volume raw_data = open_raw5(oss1.str(), oss2.str(), oss3.str(), oss4.str(), oss5.str(), width, height, depth);
		//Volume raw_data = open_raw(oss1.str(), oss2.str(), oss3.str(), width, height, depth);
	//	raw_data.draw_slice_rgb("test.bmp",10);







		ostringstream oss_in;
		oss_in << argv[2] << "/Segm_MRBrainS13_"  << std::setw(2) << std::setfill('0') << n+1 << "-input1" << ".tif";
		save_tiff(oss_in.str(), raw_data.to_vector(), data_shape, 0);

		oss_in.str("");
		oss_in.clear();
		oss_in << argv[2] << "/Segm_MRBrainS13_"  << std::setw(2) << std::setfill('0') << n+1 << "-input2" << ".tif";
		save_tiff(oss_in.str(), raw_data.to_vector(), data_shape, 1);

		oss_in.str("");
		oss_in.clear();
		oss_in << argv[2] << "/Segm_MRBrainS13_"  << std::setw(2) << std::setfill('0') << n+1 << "-input3" << ".tif";
		save_tiff(oss_in.str(), raw_data.to_vector(), data_shape, 2);


		VolumeShape stepsize = VolumeShape{data_shape.z - sub_shape.z, 1, data_shape.w - sub_shape.w, data_shape.h - sub_shape.h};

		stepsize.w = (nstep_x == 1) ? 1 : ((stepsize.w % nstep_x) ? 1 : 0) + stepsize.w / (nstep_x - 1);
		stepsize.h = (nstep_y == 1) ? 1 : ((stepsize.h % nstep_y) ? 1 : 0) + stepsize.h / (nstep_y - 1);
		stepsize.z = (nstep_z == 1) ? 1 : ((stepsize.z % nstep_z) ? 1 : 0) + stepsize.z / (nstep_z - 1);




		VolumeShape out_shape = VolumeShape{data_shape.z, net.output_shape().c, data_shape.w, data_shape.h};
		vector<float> final_output(out_shape.size());
		vector<float> final_count(out_shape.size());

		cout << net.volumes[0]->x.shape << endl;
		cout << raw_data.shape << endl;
		// net.set_input(raw_data);

		Timer wholetimer;
		bool bla(false);
		for(int z(0); z < nstep_z; z++){
			for(int y(0); y < nstep_y; y++){
				for(int x(0); x < nstep_x; x++){
					int idxx = min(x * stepsize.w, data_shape.w - sub_shape.w);
					int idxy = min(y * stepsize.h, data_shape.h - sub_shape.h);
					int idxz = min(z * stepsize.z, data_shape.z - sub_shape.z);
					copy_subvolume_test(raw_data, net.input(), idxx, idxy, idxz);
					//(*net.input().buf) *= .8;
					//net.input().draw_slice("jemoeder.png",0);

					Timer ftimer;

					//for (size_t n(0); n < 3; ++n) {
					  net.forward();
					  cout << "forward took:" << ftimer.since() << endl;

					  vector<float> net_output = net.output().to_vector();
					  vector<float>::iterator net_output_it(net_output.begin());

					  for(int z2(0); z2 < sub_shape.z; z2++) {
					    for(int c2(0); c2 < out_shape.c; c2++) {
					      for(int x2(0); x2 < sub_shape.w; x2++) {
							for(int y2(0); y2 < sub_shape.h; y2++, ++net_output_it){
							  float dist = pow(float(z2 - sub_shape.z / 2) / (sub_shape.z / 2), 2.0) +
							    pow(float(x2 - sub_shape.w / 2) / (sub_shape.w / 2), 2.0) +
							    pow(float(y2 - sub_shape.h / 2) / (sub_shape.h / 2), 2.0);
							  float weight = exp(-dist);
							  vector<float>::iterator final_output_it(final_output.begin() + out_shape.offset(idxz + z2, c2, idxx + x2, idxy + y2));
							  vector<float>::iterator final_count_it(final_count.begin() + out_shape.offset(idxz + z2, c2, idxx + x2, idxy + y2));
							  *final_output_it += *net_output_it * weight;
							  //*final_output_it += rand_float() * weight;
							  *final_count_it += weight;
							}
					      }
					    }
					  }

					  //float loss = net.calculate_loss(tiff_label);
					// }
				}
			}
		}

		vector<float>::iterator final_output_it(final_output.begin()), final_output_end(final_output.end());
		vector<float>::iterator final_count_it(final_count.begin());
		for (; final_output_it != final_output_end; ++final_output_it, ++final_count_it)
		  *final_output_it /= (*final_count_it) + .00001;
			// *final_output_it = rand_float();

		cout << "the program took (per-image):" << wholetimer.since() << endl;

		cudaDeviceSynchronize();

		ostringstream oss_label;
		oss_label << argv[2] << "/Segm_MRBrainS13_"  << std::setw(2) << std::setfill('0') << n+1 <<".raw";
		save_raw_classification(oss_label.str(), final_output, out_shape);

		for (int ch(0); ch < out_shape.c; ch++){
			ostringstream oss_label1;
			oss_label1 << argv[2] << "/Segm_MRBrainS13_"  << std::setw(2) << std::setfill('0') << n+1 << "-" << ch << ".tif";
			save_tiff(oss_label1.str(), final_output, out_shape, ch);
			ostringstream oss;
			oss << "mrbrain-test/Segm_MRBrainS13_"  << std::setw(2) << std::setfill('0') << n+1 << "-" << ch << ".tif";
			save_tiff(oss.str(), raw_data.to_vector(), data_shape, ch);
		}



	}

	// int i(0);
	// for ( int i(0); i < net.output().shape.z; i++) {
	// //while (i < 20){
	// 	ostringstream ose;
	// 	ose << argv[3] << "in-" << i << ".png";
	// 	net.input().draw_slice(ose.str(),i);

	// 	ostringstream oss;
	// 	oss << argv[3] << i << ".png";
	// 	net.output().draw_slice(oss.str(), i);

	// 	ostringstream ost;
	// 	ost << argv[3] << i << "t5.png";
	// 	net.output().draw_slice(ost.str(), i, 0.5);
	// 	// count++;
	// }
}

