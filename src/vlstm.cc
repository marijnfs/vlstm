#include "vlstm.h"

//Volume Operation
VolumeOperation::VolumeOperation(Op &op_, VolumeSet &in_, VolumeSet &out_, int dt_, bool first_) :
	op(op_), T(in.x.shape.z), dt(dt_), first(first_), 
	in(in_), out(out_), 
	in_t(in.x.create_slice_tensor()),
	out_t(out.x.create_slice_tensor()), 
	in_err_t(in.x.create_slice_tensor()),
	out_err_t(out.x.create_slice_tensor()) 
{}

void VolumeOperation::forward(int t) {
	if (dt > t)
		return;
	in_t.data = in.x.slice(t - dt);
	out_t.data = out.x.slice(t);

	op.forward(in_t, out_t);
}

void VolumeOperation::backward(int t) {
	if (dt > t)
		return;
	in_t.data = in.x.slice(t - dt);
	out_t.data = out.x.slice(t);

	in_err_t.data = in.diff.slice(t - dt);
	out_err_t.data = out.diff.slice(t - dt);

	op.backward(in_t, out_t, out_err_t, in_err_t);
}

void VolumeOperation::forward_dry_run() {
	op.forward_dry_run(in_t, out_t);
}

//Volume Operation, 2 inputs
VolumeOperation2::VolumeOperation2(Op &op_, VolumeSet &in_, VolumeSet &in2_, VolumeSet &out_, int dt_, bool first) :
	op(op_),T(in.x.shape.z), dt(dt_), in(in_), first(first_), in2(in2_), out(out_), 
	in_t(in.x.create_slice_tensor()),
	in2_t(in2.x.create_slice_tensor()),
	out_t(out.x.create_slice_tensor()), 
	in_err_t(in.x.create_slice_tensor()),
	in2_err_t(in2.x.create_slice_tensor()),
	out_err_t(out.x.create_slice_tensor()) 
{}

void VolumeOperation2::forward(int t) {
	if (dt > t)
		return;
	//we only delay the first input
	in_t.data = in.x.slice(t - dt);
	in2_t.data = in2.x.slice(t);
	out_t.data = out.x.slice(t);

	op.forward(in_t, in2_t, out_t);

	Tensor<F> &in, Tensor<F> &in2, Tensor<F> &out, Tensor<F> &out_grad, Tensor<F> &in_grad, Tensor<F> &in2_grad
}

void VolumeOperation2::backward(int t) {
	if (dt > t)
		return;
	in_t.data = in.x.slice(t - dt);
	in2_t.data = in2.x.slice(t);
	out_t.data = out.x.slice(t);

	in_err_t.data = in.diff.slice(t - dt);
	in2_err_t.data = in2.diff.slice(t);
	out_err_t.data = out.diff.slice(t);

	op.backward(in_t, in2_t, out_t, out_err_t, in_err_t, in2_err_t);
}

void VolumeOperation2::forward_dry_run() {
	op.forward_dry_run(in_t, in2_t, out_t);
}

//LSTM operation
LSTMOperation::LSTMOperation(VolumeShape in, int kg, int ko, int c) :
	T(in.z),
	ConvolutionOperation xi(in.c, c, kg, kg), hi(c, c, kg, kg); //input gate
	ConvolutionOperation xr(in.c, c, kg, kg), hr(c, c, kg, kg); //remember gate (forget gates dont make sense!)
	ConvolutionOperation xs(in.c, c, kg, kg), hs(c, c, kg, kg); //cell input
	ConvolutionOperation xo(in.c, c, ko, ko), ho(c, c, ko, ko), co(c, c, ko, ko), //output gate
	in_shape(s),
	vin(0),
	vout(0)
{
	add_volume("x", s);
	add_volume("h", VolumeShape{in.z, c, in.w, in.h});

	add_operations();

	vin = &volumes["x"];
	vout = &volumes["h"];
}

//Constructor if you already have sets
LSTMOperation(VolumeSet &in, VolumeSet &out, int kg, int ko, int c) : 
	T(in.shape.z),
	ConvolutionOperation xi(in.c, c, kg, kg), hi(c, c, kg, kg); //input gate
	ConvolutionOperation xr(in.c, c, kg, kg), hr(c, c, kg, kg); //remember gate (forget gates dont make sense!)
	ConvolutionOperation xs(in.c, c, kg, kg), hs(c, c, kg, kg); //cell input
	ConvolutionOperation xo(in.c, c, ko, ko), ho(c, c, ko, ko), co(c, c, ko, ko), //output gate
	in_shape(s),
	vin(0),
	vout(0)
{
	add_volume("x", in);
	add_volume("h", out);

	add_operations();

	vin = &volumes["x"];
	vout = &volumes["h"];
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

	add_op("fs", "fi", "c", gate)
	add_op("c", "fr", "c", gate, true)
	add_op("c", "fc", tan)

	add_op("x", "o", xo);
	add_op("h", "o", ho, true);
	add_op("c", "o", co);
	add_op("o", "fo", sig);

	add_op("fc", "fo", "h", gate);
}

void LSTMOperation::add_volume(std::string name, VolumeShape shape) {
	volumes[name] = new VolumeSet(shape);
}

void LSTMOperation::add_volume(std::string name, VolumeSet &set) {
	volumes[name] = set;
}

bool LSTMOperation::exists(std::string name) {
	return volumes.count(name);
}

VolumeShape LSTMOperation::output_shape(VolumeShape in, Operation &op) {
	TensorShape out = op.output_shape(TensorShape{1, in.c, in.w, in.h});
	return VolumeShape{in.z, out.c, out.w, out.h}
}

void LSTMOperation::add_op(string in, string out, Operation &op, bool delay) {
	VolumeSet &in(*volumes[in]);

	bool first(false);
	if (!exists(out)) {
		add_volume(out, output_shape(in, op));
		first = true;
	}
	VolumeSet &out(*volumes[out]);

	int dt = delay ? 1 : 0;
	operations.push_back(new VolumeOperation(op, in, out, dt, first));
}

void LSTMOperation::add_op(string in, string in2, string out, Operation2 &op, bool delay) {
	VolumeSet &in(*volumes[in]);
	VolumeSet &in2(*volumes[in2]);

	bool first(false);
	if (!exists(out)) {
		add_volume(out, output_shape(in, op));
		first = true;
	}
	VolumeSet &out(*volumes[out]);

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
		for (auto &op : operations)
			op->backward(i);
}

void LSTMOperation::forward_dry_run() {
	for (int i(0); i < T; ++i)
		for (auto &op : operations)
			op->forward_dry_run();
}


//V lstm
VLSTM::VLSTM(VolumeShape s, int kg, int ko, int c):
	x(s), y(VolumeShape{s.z, c, s.w, s.h}),
	x6(s), y6(VolumeShape{s.z, c, s.w, s.h}),
	operation(s, kg, ko, c)
{
	for (size_t i(0); i < 6; ++i)
		operations.push_back(LSTMOperation(x6.volumes[i], y6.volumes[i], kg, ko, c));
}

VLSTM::forward() {
	divide(x.x, x6.x);
	for (auto &op : operations)
		op.forward();
	combine(y6.x, y.x)
}

VLSTM::backward() {
	divide(y.diff, y6.diff);

	for (int i(operations.size() - 1); i > 0; --i)
		operations[i].backward();

	combine(x6.diff, x.diff);
}
