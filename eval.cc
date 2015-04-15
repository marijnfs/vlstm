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
	if (argc < 4) {
		cout << "usage: eval [net] [in] [out]" << endl;
		return 1;
	}

	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);

	ostringstream oss;

	Volume tiff_data = open_tiff(argv[2], true);
	cout << tiff_data.shape << endl;


	//VolumeShape shape = tiff_data.shape;
	//VolumeNetwork net(shape);

	VolumeShape data_shape = tiff_data.shape;
	VolumeShape sub_shape = tiff_data.shape;
	int nstep_x = 2;
	int nstep_y = 2;
	int nstep_z = 2;

	// Volume in_data(shape);
	// Volume out_data(shape);

	// in_data.init_normal(0, .5);
	// out_data.init_normal(0, .5);
	sub_shape.w = 256;
	sub_shape.h = 256;
	sub_shape.z = 25;

	VolumeShape stepsize = VolumeShape{data_shape.z - sub_shape.z, 1, data_shape.w - sub_shape.w, data_shape.h - sub_shape.h};
	stepsize.w = ((stepsize.w % nstep_x) ? 1 : 0) + stepsize.w / (nstep_x - 1);
	stepsize.h = ((stepsize.h % nstep_y) ? 1 : 0) + stepsize.h / (nstep_y - 1);
	stepsize.z = ((stepsize.z % nstep_z) ? 1 : 0) + stepsize.z / (nstep_z - 1);

	//We need a volume for sub targets
	VolumeNetwork net(sub_shape);


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
	net.add_fc(16);
	net.add_vlstm(7, 7, 16);
	net.add_fc(25);
	net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_fc(45);
	net.add_tanh();
	net.add_vlstm(7, 7, 64);
	//net.add_fc(32);
	net.add_tanh();
	net.add_fc(2);
	net.add_softmax();

	net.finish();
	//net.init_normal(0, .1);
	net.load(argv[1]);
	//net.init_uniform(.1);


	vector<float> final_output(data_shape.z * net.output_shape().c * data_shape.w * data_shape.h);
	vector<int> final_count(data_shape.z * net.output_shape().c * data_shape.w * data_shape.h);

	cout << net.volumes[0]->x.shape << endl;
	cout << tiff_data.shape << endl;
	// net.set_input(tiff_data);

	for(int z(0); z < nstep_z; z++){
		for(int y(0); y < nstep_y; y++){
			for(int x(0); x < nstep_x; x++){
				int idxx = min(x * stepsize.w, data_shape.w - sub_shape.w);
				int idxy = min(y * stepsize.h, data_shape.h - sub_shape.h);
				int idxz = min(z * stepsize.z, data_shape.z - sub_shape.z);
				copy_subvolume_test(tiff_data, net.input(), idxx, idxy, idxz);

				Timer ftimer;
				net.forward();
				cout << "forward took:" << ftimer.since() << endl;

				vector<float> net_output = net.output().to_vector();
				vector<float>::iterator net_output_it(net_output.begin());

				for(int z2(0); z2 < sub_shape.z; z2++) {
					for(int c2(0); c2 < sub_shape.c; c2++) {
						for(int x2(0); x2 < sub_shape.w; x2++) {
							for(int y2(0); y2 < sub_shape.h; y2++, ++net_output_it){
								vector<float>::iterator final_output_it(final_output.begin() + data_shape.offset(idxz + z2, c2, idxy + y2, idxx + x2));
								vector<int>::iterator final_count_it(final_count.begin() + data_shape.offset(idxz + z2, c2, idxy + y2, idxx + x2));
								*final_output_it += *net_output_it;
								*final_count_it += 1;
							}
						}
					}
				}

				//float loss = net.calculate_loss(tiff_label);
			}
		}
	}

	vector<float>::iterator final_output_it(final_output.begin()), final_output_end(final_output.end());
	vector<int>::iterator final_count_it(final_count.begin());
	for (; final_output_it != final_output_end; ++final_output_it, ++final_count_it)
		*final_output_it /= *final_count_it;

	cudaDeviceSynchronize();

	save_tiff("result.tiff", final_output, data_shape);

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

