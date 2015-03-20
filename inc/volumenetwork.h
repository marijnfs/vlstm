#ifndef __VOLUMENETWORK_H__
#define __VOLUMENETWORK_H__

#include "volume.h"
#include "volumeoperation.h"

struct VolumeNetwork {
	VolumeNetwork(VolumeShape shape);

	void forward();
	void backward();
	void forward_dry_run();

	void finish();

	void set_input(Volume &in);
	float calculate_loss(Volume &target);
	void update(float lr);
	void clear();
	void init_normal(float mean, float std);

	Volume &output();

	void add_vlstm(int kg, int ko, int c);
	void add_fc(int c);
	void add_softmax();
	void add_tanh();
	void add_sigmoid();

	std::vector<Parametrised<F>*> params;

	std::vector<VolumeOperation*> operations;
	std::vector<VolumeSet*> volumes;
	std::vector<VolumeShape> shapes;
};

#endif