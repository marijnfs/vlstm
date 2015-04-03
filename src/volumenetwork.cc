#include "volumenetwork.h"
#include "volumeoperation.h"
#include "vlstm.h"

using namespace std;

VolumeNetwork::VolumeNetwork(VolumeShape shape) : n_params(0) {
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
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

	register_params();
	param.resize(n_params);
	grad.resize(n_params);

	align_params();
	for(auto o : operations)
		o->sharing();


	//init_normal(0, 0);
	a.resize(param.n);
	b.resize(param.n);
	c.resize(param.n);
	d.resize(param.n);
	e.resize(param.n);
	rmse.resize(param.n);
	rmse += .01;
}

void VolumeNetwork::register_params() {
	for (auto &o : operations)
		o->register_params(params, grads);

 	n_params = 0;
	for (auto &p : params)
		n_params += p.n;
}

void VolumeNetwork::align_params() {

	cout << "n params: " << n_params << endl;
	//throw "";

	for (auto &p : params)
		cudaFree(*(p.ptr));
	for (auto &g : grads)
		cudaFree(*(g.ptr));

	float *ptr = param.data;
	for (auto &p : params) {
		*(p.ptr) = ptr;
		ptr += p.n;
	}

	ptr = grad.data;
	for (auto &g : grads) {
		*(g.ptr) = ptr;
		ptr += g.n;
	}

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

void VolumeNetwork::init_uniform(float var) {
	for (auto &o : operations)
		o->init_uniform(var);
}

void VolumeNetwork::set_input(Volume &in) {
	volumes[0]->x.from_volume(in);
}

Volume &VolumeNetwork::output() {
	return last(volumes)->x;
}

Volume &VolumeNetwork::input() {
	return first(volumes)->x;
}

float VolumeNetwork::calculate_loss(Volume &target) {
	last(volumes)->diff.from_volume(target);
	last(volumes)->diff -= last(volumes)->x;

	float norm = last(volumes)->diff.norm();
	return norm;
}

void VolumeNetwork::add_vlstm(int kg, int ko, int c) {
	cout << "adding vlstm" << endl;
	//cout << "adding: " << last(shapes) << " " << shape << endl;

	auto vlstm = new VLSTMOperation(last(shapes), kg, ko, c, vsm);
	auto shape = vlstm->output_shape(last(shapes));

	operations.push_back(vlstm);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_fc(int c) {
	cout << "adding fc" << endl;
	auto fc = new FCVolumeOperation(last(shapes), last(shapes).c, c);
	auto shape = fc->output_shape(last(shapes));
	operations.push_back(fc);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_softmax() {
	cout << "adding softmax" << endl;
	auto softmax = new SoftmaxVolumeOperation(last(shapes));
	auto shape = softmax->output_shape(last(shapes));
	operations.push_back(softmax);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_tanh() {
	cout << "adding tanh" << endl;
	auto tan = new TanhVolumeOperation(last(shapes));
	auto shape = tan->output_shape(last(shapes));
	operations.push_back(tan);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_sigmoid() {
	cout << "adding sigmoid" << endl;
	auto sig = new SigmoidVolumeOperation(last(shapes));
	auto shape = sig->output_shape(last(shapes));
	operations.push_back(sig);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::save(std::string path) {
	ofstream of(path, ios::binary);
	vector<float> data = param.to_vector();
	byte_write_vec(of, data);
}

void VolumeNetwork::load(std::string path) {
	ifstream in(path, ios::binary);
	vector<float> data = byte_read_vec<float>(in);
	param.from_vector(data);
}
