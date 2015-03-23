#include "vlstm.h"
#include "divide.h"

using namespace std;

//LSTM operation
LSTMOperation::LSTMOperation(VolumeShape in, int kg, int ko, int c, VolumeSetMap *reuse) :
	T(in.z),
	xi(in.c, c, kg, kg), hi(c, c, kg, kg), //input gate
	xr(in.c, c, kg, kg), hr(c, c, kg, kg), //remember gate (forget gates dont make sense!)
	xs(in.c, c, kg, kg), hs(c, c, kg, kg), //cell input
	xo(in.c, c, ko, ko), ho(c, c, ko, ko), co(c, c, ko, ko), //output gate
	in_shape(in),
	vin(0),
	vout(0)
{
	add_volume("x", VolumeShape{in.z, in.c, in.w, in.h}, reuse);
	add_volume("h", VolumeShape{in.z, c, in.w, in.h}, reuse);

	add_operations(reuse);

	vin = volumes["x"];
	vout = volumes["h"];

	//xr.bias.init_normal(1.0, 0.0);
	//xi.bias.init_normal(1.0, 0.0);
}


void LSTMOperation::add_operations(VolumeSetMap *reuse) {
	bool DELAY(true), NOW(false);
	add_op("x", "i", xi, NOW, reuse);
	add_op("h", "i", hi, DELAY, reuse);
	add_op("i", "fi", sig, NOW, reuse);

	add_op("x", "r", xr, NOW, reuse);
	add_op("h", "r", hr, DELAY, reuse);
	add_op("r", "fr", sig, NOW, reuse);

	add_op("x", "s", xs, NOW, reuse);
	add_op("h", "s", hs, DELAY, reuse);
	add_op("s", "fs", tan, NOW, reuse);

	add_op("fs", "fi", "c", gate, NOW, reuse);
	add_op("c", "fr", "c", gate, DELAY, reuse);
	// add_op("c", "fc", tan, reuse);
	add_op("c", "fc", sig, NOW, reuse);

	add_op("x", "o", xo, NOW, reuse);
	add_op("h", "o", ho, DELAY, reuse);
	//add_op("c", "o", co, reuse);
	add_op("o", "fo", sig, NOW, reuse);

	add_op("fc", "fo", "h", gate, NOW, reuse);
}

void LSTMOperation::add_volume(string name, VolumeShape shape, VolumeSetMap *reuse) {
	if (reuse)
		volumes[name] = new VolumeSet(shape, *(*reuse)[name]);
	else
		volumes[name] = new VolumeSet(shape);
}

void LSTMOperation::add_volume(string name, VolumeSet &set) {
	volumes[name] = &set;
}

bool LSTMOperation::exists(string name) {
	return volumes.count(name);
}

void LSTMOperation::init_normal(F mean, F std) {
	for (auto &p : parameters)
		p->init_normal(mean, std);
}

void LSTMOperation::register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &grads) {
	for (auto &p : parameters)
		p->register_params(params, grads);
}

void LSTMOperation::update(float lr) {
	for (auto &p : parameters) {
		p->update(lr);
		//cout << p->grad_to_vector() << endl;
	}

}

void LSTMOperation::clear() {
	for (auto& v : volumes) {
		 // if (v.first != "x" && v.first != "h")
		 	v.second->x.zero();
		//v.second->x.zero();
		v.second->diff.zero();
	}
	for (auto& p : parameters)
		p->zero_grad();
}

void LSTMOperation::add_op(string ins, string outs, Operation<F> &op, bool delay, VolumeSetMap *reuse) {
	VolumeSet &in(*volumes[ins]);

	bool first(false);
	if (!exists(outs)) {
		add_volume(outs, output_shape(in.shape, op), reuse);
		first = true;
	}
	VolumeSet &out(*volumes[outs]);

	int dt = delay ? 1 : 0;

	operations.push_back(new TimeOperation1(op, in, out, dt, first));
	try {
		parameters.push_back(&dynamic_cast<Parametrised<F> &>(op));
		cout << "a parameter" << endl;
	} catch (const std::bad_cast& e) {
		cout << "not a parameter" << endl;
	}
}

void LSTMOperation::add_op(string ins, string in2s, string outs, Operation2<F> &op, bool delay, VolumeSetMap *reuse) {
	VolumeSet &in(*volumes[ins]);
	VolumeSet &in2(*volumes[in2s]);

	bool first(false);
	if (!exists(outs)) {
		add_volume(outs, output_shape(in.shape, op), reuse);
		first = true;
	}
	VolumeSet &out(*volumes[outs]);

	int dt = delay ? 1 : 0;
	operations.push_back(new TimeOperation2(op, in, in2, out, dt, first));
}


void LSTMOperation::forward() {
//	int T(in_shape.z);
	for (int t(0); t < T; ++t)
		for (auto &op : operations) {
			op->forward(t);
		}

}

void LSTMOperation::backward() {
	// cout << "back" << endl;
	for (int t(T - 1); t > 0; --t)
		for (int n(operations.size() - 1); n >= 0; --n) {
			operations[n]->backward(t);
		}
	// cout << "scaling" << endl;
	for (auto &p : parameters)
		//p->scale_grad(1.0 / (in_shape.z * in_shape.w * in_shape.h));
		p->scale_grad(1.0 / sqrt(in_shape.z * in_shape.w * in_shape.h));
	// cout << "done" << endl;
}

void LSTMOperation::forward_dry_run() {
	for (auto &op : operations)
		op->forward_dry_run();
}


//Vlstm
VLSTMOperation::VLSTMOperation(VolumeShape s, int kg, int ko, int c_) : c(c_)
{
	// for (size_t i(0); i < 6; ++i)
		// operations.push_back(new LSTMOperation(*(x6.volumes[i]), *(y6.volumes[i]), kg, ko, c));

	operations.push_back(new LSTMOperation(VolumeShape{s.z, s.c, s.w, s.h}, kg, ko, c));
	operations.push_back(new LSTMOperation(VolumeShape{s.z, s.c, s.w, s.h}, kg, ko, c, &(operations[0]->volumes)));

	operations.push_back(new LSTMOperation(VolumeShape{s.w, s.c, s.z, s.h}, kg, ko, c, &(operations[0]->volumes)));
	operations.push_back(new LSTMOperation(VolumeShape{s.w, s.c, s.z, s.h}, kg, ko, c, &(operations[0]->volumes)));

	operations.push_back(new LSTMOperation(VolumeShape{s.h, s.c, s.w, s.z}, kg, ko, c, &(operations[0]->volumes)));
	operations.push_back(new LSTMOperation(VolumeShape{s.h, s.c, s.w, s.z}, kg, ko, c, &(operations[0]->volumes)));

	// 	volumes.push_back(new VolumeSet(VolumeShape{s.z, s.c, s.w, s.h}));
	// 	volumes.push_back(new VolumeSet(VolumeShape{s.z, s.c, s.w, s.h}));

	// 	volumes.push_back(new VolumeSet(VolumeShape{s.w, s.c, s.z, s.h}));
	// 	volumes.push_back(new VolumeSet(VolumeShape{s.w, s.c, s.z, s.h}));

	// 	volumes.push_back(new VolumeSet(VolumeShape{s.h, s.c, s.w, s.z}));
	// 	volumes.push_back(new VolumeSet(VolumeShape{s.h, s.c, s.w, s.z}));

	clear();
	for (auto &op : operations)
		op->forward_dry_run();
}

void VLSTMOperation::clear() {
	for (auto& o : operations) {
		o->clear();
	}
}

void VLSTMOperation::forward(Volume &in, Volume &out) {
	for (size_t i(0); i < operations.size(); ++i) {
		operations[i]->clear();
		divide(in, operations[i]->input().x, i);
		operations[i]->forward();
		combine(operations[i]->output().x, out, i);
		if (i == 0) {
			operations[0]->volumes["c"]->x.draw_slice("c1.png", 1);
			operations[0]->volumes["c"]->x.draw_slice("c8.png", 8);
			operations[0]->volumes["i"]->x.draw_slice("i1.png", 1);
			operations[0]->volumes["i"]->x.draw_slice("i8.png", 8);
			operations[0]->volumes["o"]->x.draw_slice("o1.png", 1);
			operations[0]->volumes["o"]->x.draw_slice("o8.png", 8);
		}
	}
}

void VLSTMOperation::backward(VolumeSet &in, VolumeSet &out) {
	for (size_t i(0); i < operations.size(); ++i) {
		//forward
		operations[i]->clear();
		divide(in.x, operations[i]->input().x, i);
		operations[i]->forward();

		//backward
		divide(out.diff, operations[i]->output().diff, i);
		operations[i]->backward();
		combine(operations[i]->input().diff, in.diff, i);
	}
}

void VLSTMOperation::forward_dry_run(Volume &in, Volume &out){

}

VolumeShape VLSTMOperation::output_shape(VolumeShape s) {
	return VolumeShape{s.z, c, s.w, s.h};
}


void VLSTMOperation::init_normal(F mean, F std) {
	for (auto &o : operations)
		o->init_normal(mean, std);
}

void VLSTMOperation::register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &grads) {
	for (auto &o : operations)
		o->register_params(params, grads);
}

void VLSTMOperation::update(float lr) {
	for (auto &o : operations) {
		// cout << "update lstm op" << endl;
		o->update(lr);
	}
}
