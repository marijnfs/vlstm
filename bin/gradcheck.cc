#include <iostream>
#include <cuda.h>
#include <sstream>

#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
#include "tiff.h"
#include "log.h"

using namespace std;

void print_last(vector<float> vals, int n) {
	for (size_t i(vals.size() - n); i < vals.size(); ++i)
		cout << vals[i] << " ";
	cout << endl;
}

int main(int argc, char **argv) {

	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);

	ostringstream oss;

	Volume in_data(VolumeShape{5, 1, 5, 5});

	Volume out_data_tmp(VolumeShape{5, 3, 5, 5});
	Volume out_data(VolumeShape{5, 3, 5, 5});
	out_data_tmp.init_normal(0.1, .4);
	SoftmaxVolumeOperation op(out_data_tmp.shape);
	op.forward(out_data_tmp, out_data);
	cout << out_data.to_vector() << endl;
	// return 1;

	in_data.init_normal(0.1, .4);//Volume tiff_label = open_tiff(, false, true);

	VolumeShape shape = in_data.shape;

	VolumeNetwork net(shape);

	net.add_fc(2);
	net.add_sigmoid();
	net.add_vlstm(1, 1, 3);
	net.add_softmax();
	net.finish();

	net.init_uniform(.2);

	net.set_input(in_data);
	//net.volumes[0]->x.draw_slice("in_3.png", 3);
	//tiff_label.draw_slice("label_3.png", 3);

	CudaVec vec;
	CudaVec grad;
	vec = net.param;
	net.forward();
	net.calculate_loss(out_data);
	net.backward();

	grad = net.grad;
	vector<float> grad_vec = grad.to_vector();

	float eps = .01;
	for (int i = 0; i < net.param.n; i++){
		// cout << "p:" << net.param.to_vector() << endl;
		net.param.add(i, eps);
		// cout << "p+:" <<  net.param.to_vector() << endl;
		net.forward();
		// cout << "== + ===" << endl;
		// for (auto &v : net.volumes)
		// 	cout << v->x.to_vector() << endl;
		// cout << out_data.to_vector() << endl;

		float loss_plus = net.calculate_loss(out_data);
		// cout << "===== " << loss_plus << endl;
		net.param.add(i, -2.0*eps);
		// cout << "p-:" <<  net.param.to_vector() << endl;
		net.forward();
		// cout << "== - ===" << endl;
		// for (auto &v : net.volumes)
		// 	cout << v->x.to_vector() << endl;
		// cout << out_data.to_vector() << endl;


		float loss_min = net.calculate_loss(out_data);
		// cout << "===== " << loss_min << endl;
		float check = ((loss_plus - loss_min) / (2.0 * eps));
		cout << i << " / " << grad_vec[i] << " " << check << " " << fabs(grad_vec[i] / check) << endl;
		net.param = vec;
	}
	cudaDeviceSynchronize();

}

