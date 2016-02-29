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
	align_params();
	for(auto o : operations)
		o->sharing();
	clear();

	//init_normal(0, 0);
	// a.resize(param.n);
	// b.resize(param.n);
	// c.resize(param.n);
	// d.resize(param.n);
	// e.resize(param.n);
	// rmse.resize(param.n);
	// rmse += .01;
}

void VolumeNetwork::register_params() {
	for (auto &o : operations)
		o->register_params(params, fast_params, grads, fast_grads);

 	n_params = 0;
	for (auto &p : params)
		n_params += p.n;

	n_fast_params = 0;
	for (auto &p : fast_params)
		n_fast_params += p.n;
}

void VolumeNetwork::align_params() {
	param.resize(n_params);
	grad.resize(n_params);

	fast_param.resize(n_fast_params);
	fast_grad.resize(n_fast_params);

	for (auto &p : params)
		cudaFree(*(p.ptr));
	for (auto &p : fast_params)
		cudaFree(*(p.ptr));
	for (auto &g : grads)
		cudaFree(*(g.ptr));

	for (auto &g : fast_grads) {
		// cout << "ptr: " << g.ptr << " " << g.n << " " << fast_grads.size() << endl;
		cudaFree(*(g.ptr));
	}

	position_params(param.data, fast_param.data, grad.data, fast_grad.data);
	cout << "n params: " << n_params << endl;
	//throw "";
}

void VolumeNetwork::position_params(float *pos_param, float *pos_fast_param, float *pos_grad, float *pos_fast_grad) {
	float *ptr = pos_param;
	for (auto &p : params) {
		*(p.ptr) = ptr;
		ptr += p.n;
	}

	ptr = pos_fast_param;
	for (auto &p : fast_params) {
		*(p.ptr) = ptr;
		ptr += p.n;
	}

	ptr = pos_grad;
	for (auto &g : grads) {
		*(g.ptr) = ptr;
		ptr += g.n;
	}

	ptr = pos_fast_grad;
	for (auto &g : fast_grads) {
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
		// if (!i)
		// 	volumes[i]->diff.zero();
		// else
		// 	volumes[i]->zero();
		if (i)
			volumes[i]->x.zero();
		volumes[i]->diff.zero();
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
	first(volumes)->x.from_volume(in);
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

	float norm = last(volumes)->diff.norm2() * 0.5;
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

void VolumeNetwork::add_univlstm(int kg, int ko, int c) {
	cout << "adding unidirectional vlstm" << endl;
	//cout << "adding: " << last(shapes) << " " << shape << endl;

	auto vlstm = new UniVLSTMOperation(last(shapes), kg, ko, c, vsm);
	auto shape = vlstm->output_shape(last(shapes));

	operations.push_back(vlstm);
	shapes.push_back(shape);
	volumes.push_back(new VolumeSet(shape));
}

void VolumeNetwork::add_fc(int c, float dropout) {
	cout << "adding fc" << endl;
	auto fc = new FCVolumeOperation(last(shapes), last(shapes).c, c, dropout);
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

void VolumeNetwork::describe(std::ostream &out) {
	for (auto &o : operations) {
		o->describe(out);
		out << endl;
	}
	out.flush();
}

void VolumeNetwork::set_fast_weights(Tensor<float> &weights) {
	//weights come out of network ordered weights first and then by time, while they are packed reversely
	//we have to account for that here
	int T = input().shape.z;

	for (size_t t(0); t < T; ++t) {
		int shift(0);
		for (CudaPtr<F> &param : fast_params) {
			int n = param.n / T;
			F *dest = (*param.ptr) + t * n;
			F *src = weights.ptr() + t * weights.c + shift;

			cout << dest << " " << src << " " << n << " " << weights.c << " " << weights.h << endl;
			copy_gpu_to_gpu<>(src, dest, n);
			shift += n;
		}
	}
}

void VolumeNetwork::get_fast_grads(Tensor<float> &grad_vec) {
	//weights come out of network ordered weights first and then by time, while they are packed reversely
	//we have to account for that here
	int T = input().shape.z;

	for (size_t t(0); t < T; ++t) {
		int shift(0);
		for (CudaPtr<F> &grad : fast_grads) {
			int n = grad.n / T;
			F *src = (*grad.ptr) + t * n;
			F *dest = grad_vec.ptr() + t * grad_vec.c + shift;

			copy_gpu_to_gpu<>(src, dest, n);
			shift += n;
		}
	}
}
