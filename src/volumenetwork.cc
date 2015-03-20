#include "volumenetwork.h"
#include "volumeoperation.h"
#include "vlstm.h"

using namespace std;

VolumeNetwork::VolumeNetwork(VolumeShape shape) {
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
	forward_dry_run();
}

void VolumeNetwork::forward() {
	clear();
	for (int i(0); i < operations.size(); i++) {
		operations[i]->forward(volumes[i]->x, volumes[i+1]->x);
	}
}

void VolumeNetwork::backward() {
	for (int i(operations.size() - 1); i >= 0; i--) {
		operations[i]->backward(*volumes[i], *volumes[i+1]);
		operations[i]->backward_weights(*volumes[i], *volumes[i+1]);
	}
}

void VolumeNetwork::forward_dry_run() {
	for (int i(0); i < operations.size(); i++) {
		operations[i]->forward_dry_run(volumes[i]->x, volumes[i+1]->x);
	}
}

void VolumeNetwork::finish() {
	forward_dry_run();
}

void VolumeNetwork::update(float lr) {
	for (int i(0); i < operations.size(); i++)
		operations[i]->update(lr);
}

void VolumeNetwork::clear() {
	for (int i(0); i < volumes.size(); i++) {
		if (!i)
			volumes[i]->diff.zero();
		else
			volumes[i]->zero();
	}
}

void VolumeNetwork::init_normal(float mean, float std) {
	for (auto &o : operations)
		o->init_normal(mean, std);
}

void VolumeNetwork::set_input(Volume &in) {
	volumes[0]->x.from_volume(in);
}

Volume &VolumeNetwork::output() {
	return last(volumes)->x;
}

float VolumeNetwork::calculate_loss(Volume &target) {
	last(volumes)->diff.from_volume(target);
	last(volumes)->diff -= last(volumes)->x;

	float norm = last(volumes)->diff.norm();
	return norm;
}

void VolumeNetwork::add_vlstm(int kg, int ko, int c) {
	//cout << "adding: " << last(shapes) << " " << shape << endl;

	auto vlstm = new VLSTMOperation(last(shapes), kg, ko, c);
	auto shape = vlstm->output_shape(last(shapes));

	operations.push_back(vlstm);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_fc(int c) {
	auto fc = new FCVolumeOperation(last(shapes), last(shapes).c, c);
	auto shape = fc->output_shape(last(shapes));
	operations.push_back(fc);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_softmax() {
	auto softmax = new SoftmaxVolumeOperation(last(shapes));
	auto shape = softmax->output_shape(last(shapes));
	operations.push_back(softmax);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_tanh() {
	auto tan = new TanhVolumeOperation(last(shapes));
	auto shape = tan->output_shape(last(shapes));
	operations.push_back(tan);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_sigmoid() {
	auto sig = new SigmoidVolumeOperation(last(shapes));
	auto shape = sig->output_shape(last(shapes));
	operations.push_back(sig);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}
