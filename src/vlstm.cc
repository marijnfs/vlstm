#include "vlstm.h"
#include "divide.h"

using namespace std;

//LSTM operation
LSTMOperation::LSTMOperation(VolumeShape in, int kg, int ko, int c) :
	T(in.z),
	xi(in.c, c, kg, kg), hi(c, c, kg, kg), //input gate
	xr(in.c, c, kg, kg), hr(c, c, kg, kg), //remember gate (forget gates dont make sense!)
	xs(in.c, c, kg, kg), hs(c, c, kg, kg), //cell input
	xo(in.c, c, ko, ko), ho(c, c, ko, ko), co(c, c, ko, ko), //output gate
	in_shape(in),
	vin(0),
	vout(0)
{
	add_volume("x", in);
	add_volume("h", VolumeShape{in.z, c, in.w, in.h});

	add_operations();

	vin = volumes["x"];
	vout = volumes["h"];
}

//Constructor if you already have sets
LSTMOperation::LSTMOperation(VolumeSet &in, VolumeSet &out, int kg, int ko, int c) :
	T(in.shape.z),
	xi(in.shape.c, c, kg, kg), hi(c, c, kg, kg), //input gate
	xr(in.shape.c, c, kg, kg), hr(c, c, kg, kg), //remember gate (forget gates dont make sense!)
	xs(in.shape.c, c, kg, kg), hs(c, c, kg, kg), //cell input
	xo(in.shape.c, c, ko, ko), ho(c, c, ko, ko), co(c, c, ko, ko), //output gate
	in_shape(in.shape),
	vin(0),
	vout(0)
{
	add_volume("x", in);
	add_volume("h", out);

	add_operations();

	vin = volumes["x"];
	vout = volumes["h"];
}

void LSTMOperation::add_operations() {
	add_op("x", "i", xi);
	add_op("h", "i", hi, true);
	add_op("i", "fi", sig);

	add_op("x", "r", xr);
	add_op("h", "r", hr, true);
	add_op("r", "fr", sig);

	add_op("x", "s", xi);
	add_op("h", "s", hi, true);
	add_op("s", "fs", tan);

	add_op("fs", "fi", "c", gate);
	add_op("c", "fr", "c", gate, true);
	add_op("c", "fc", tan);

	add_op("x", "o", xo);
	add_op("h", "o", ho, true);
	add_op("c", "o", co);
	add_op("o", "fo", sig);

	add_op("fc", "fo", "h", gate);
}

void LSTMOperation::add_volume(string name, VolumeShape shape) {
	volumes[name] = new VolumeSet(shape);
}

void LSTMOperation::add_volume(string name, VolumeSet &set) {
	volumes[name] = &set;
}

bool LSTMOperation::exists(string name) {
	return volumes.count(name);
}

VolumeShape LSTMOperation::output_shape(VolumeShape in, Operation<F> &op) {
	TensorShape out = op.output_shape(TensorShape{1, in.c, in.w, in.h});
	return VolumeShape{in.z, out.c, out.w, out.h};
}

VolumeShape LSTMOperation::output_shape(VolumeShape in, Operation2<F> &op) {
	TensorShape out = op.output_shape(TensorShape{1, in.c, in.w, in.h});
	return VolumeShape{in.z, out.c, out.w, out.h};
}

void LSTMOperation::init_normal(F mean, F std) {
	for (auto &p : parameters)
		p->init_normal(mean, std);
}

void LSTMOperation::update(float lr) {
	for (auto &p : parameters) {
		p->update(lr);
		//cout << p->grad_to_vector() << endl;
	}

}

void LSTMOperation::clear() {
	for (auto& v : volumes) {
		if (v.first != "x")
			v.second->x.zero();
		v.second->diff.zero();
	}
	for (auto& p : parameters)
		p->zero_grad();
}

void LSTMOperation::add_op(string ins, string outs, Operation<F> &op, bool delay) {
	VolumeSet &in(*volumes[ins]);

	bool first(false);
	if (!exists(outs)) {
		add_volume(outs, output_shape(in.shape, op));
		first = true;
	}
	VolumeSet &out(*volumes[outs]);

	int dt = delay ? 1 : 0;

	operations.push_back(new VolumeOperation(op, in, out, dt, first));
	try {
		parameters.push_back(&dynamic_cast<Parametrised<F> &>(op));
		cout << "a parameter" << endl;
	} catch (const std::bad_cast& e) {
		cout << "not a parameter" << endl;
	}
}

void LSTMOperation::add_op(string ins, string in2s, string outs, Operation2<F> &op, bool delay) {
	VolumeSet &in(*volumes[ins]);
	VolumeSet &in2(*volumes[in2s]);

	bool first(false);
	if (!exists(outs)) {
		add_volume(outs, output_shape(in.shape, op));
		first = true;
	}
	VolumeSet &out(*volumes[outs]);

	int dt = delay ? 1 : 0;
	operations.push_back(new VolumeOperation2(op, in, in2, out, dt, first));
}

void LSTMOperation::forward() {
	int T(in_shape.z);

	for (int i(0); i < T; ++i)
		for (auto &op : operations)
			op->forward(i);
}

void LSTMOperation::backward() {
	for (int i(T - 1); i > 0; --i)
		for (int n(operations.size() - 1); n >= 0; --n)
			operations[n]->backward(i);
}

void LSTMOperation::forward_dry_run() {
	for (auto &op : operations)
		op->forward_dry_run();
}


//V lstm
VLSTM::VLSTM(VolumeShape s, int kg, int ko, int c):
	x(s), y(VolumeShape{s.z, c, s.w, s.h}),
	x6(s),
	y6(VolumeShape{s.z, c, s.w, s.h})
{
	for (size_t i(0); i < 6; ++i)
		operations.push_back(new LSTMOperation(*(x6.volumes[i]), *(y6.volumes[i]), kg, ko, c));

	clear();
	for (auto &op : operations)
		op->forward_dry_run();
}

void VLSTM::clear() {
	for (auto& o : operations) {
		o->clear();
	}

	//x.clear();
	y.zero();
	x6.zero();
	y6.zero();
}

void VLSTM::forward() {
	divide(x.x, x6.x);
	/*
	x6.x[0]->zero();
	x6.x[1]->zero();
	x6.x[2]->zero();
	x6.x[3]->zero();
	x6.x[4]->zero();
	x6.x[5]->zero();
	//x6.x[0]->draw_slice("x0.png", 3);
	//x6.x[1]->draw_slice("x1.png", 3);
	//x6.x[2]->draw_slice("x2.png", 3);
	//x6.x[3]->draw_slice("x3.png", 3);
	//x6.x[4]->draw_slice("x4.png", 3);
	//x6.x[5]->draw_slice("x5.png", 3);

	x.x.zero();
	x.x.draw_slice("xold.png", 3);
	combine(x6.x, x.x);
	x.x.draw_slice("xnew.png", 3);
	*/

	for (size_t n(2); n < 4; ++n)
		operations[n]->forward();

	//for (auto &op : operations)
	//		op->forward();
	for (auto &x : y6.x)
		cout << x->norm() << endl;
	combine(y6.x, y.x);

}

void VLSTM::backward() {
	divide(y.diff, y6.diff);

	for (size_t n(2); n < 4; ++n)
		operations[n]->backward();
	
	//for (auto &o : operations)
	//		o->backward();

	combine(x6.diff, x.diff);
}

void VLSTM::init_normal(F mean, F std) {
	for (auto &o : operations)
		o->init_normal(mean, std);
}

void VLSTM::update(float lr) {
	for (auto &o : operations) {
		cout << "update lstm op" << endl;
		o->update(lr);
	}
}
