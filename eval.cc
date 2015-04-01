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
	if (argc < 4) {
		cout << "usage: eval [net] [in] [out]" << endl;
		return 1;
	}

	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);

	ostringstream oss;

	Volume tiff_data = open_tiff(argv[2], true);
	//Volume tiff_label = open_tiff(, false, true);

	cout << tiff_data.shape << endl;
	//cout << tiff_label.shape << endl;

	VolumeShape shape = tiff_data.shape;

	VolumeNetwork net(shape);

	net.add_fc(8);
	//net.add_tanh();

	net.add_vlstm(7, 7, 8);
	net.add_fc(10);
	net.add_tanh();
	net.add_vlstm(7, 7, 16);
	net.add_fc(32);
	net.add_tanh();
	/*net.add_vlstm(3, 5, 16);
	net.add_fc(16);
	net.add_tanh();*/
	net.add_fc(2);
	//net.add_sigmoid();
	net.add_softmax();
	// net.add_tanh();
	// net.add_tanh();
	// net.add_fc(1);

	net.finish();
	//net.init_normal(0, .1);
	net.load(argv[1]);

	cout << net.volumes[0]->x.shape << endl;
	cout << tiff_data.shape << endl;
	net.set_input(tiff_data);
	net.volumes[0]->x.draw_slice("in_3.png", 3);
	//tiff_label.draw_slice("label_3.png", 3);




	Timer ftimer;
	net.forward();
	cout << "forward took:" << ftimer.since() << endl;


	//float loss = net.calculate_loss(tiff_label);

	cudaDeviceSynchronize();

	for ( int i(0); i < net.output().shape.z; i++) {
		ostringstream oss;
		oss << argv[3] << i << ".png";
		net.output().draw_slice(oss.str(), i);


	}
}

