#include "volumeoperation.h"
#include "vlstm.h"

using namespace std;


//Volume Operation
VolumeOperation::VolumeOperation(Operation<F> &op_, VolumeSet &in_, VolumeSet &out_, int dt_, bool first_) :
	op(op_), T(in_.x.shape.z), dt(dt_), first(first_),
	in(in_), out(out_),
	in_t(in_.x.slice_shape(), 0),
	out_t(out_.x.slice_shape(), 0),
	in_err_t(in_.x.slice_shape(), 0),
	out_err_t(out_.x.slice_shape(), 0)
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
	out_err_t.data = out.diff.slice(t);

	op.backward(in_t, out_t, out_err_t, in_err_t, 1.0);
	op.backward_weights(in_t, out_err_t, 1.0);
}

void VolumeOperation::forward_dry_run() {
	op.forward_dry_run(in_t, out_t);
}

//Volume Operation, 2 inputs
VolumeOperation2::VolumeOperation2(Operation2<F> &op_, VolumeSet &in_, VolumeSet &in2_, VolumeSet &out_, int dt_, bool first_) :
	op(op_),T(in_.x.shape.z), dt(dt_), in(in_), first(first_), in2(in2_), out(out_),
	in_t(in_.x.slice_shape(), 0),
	in2_t(in2_.x.slice_shape(), 0),
	out_t(out_.x.slice_shape(), 0),
	in_err_t(in_.x.slice_shape(), 0),
	in2_err_t(in2_.x.slice_shape(), 0),
	out_err_t(out_.x.slice_shape(), 0)
{}

void VolumeOperation2::forward(int t) {
	if (dt > t)
		return;
	//we only delay the first input
	in_t.data = in.x.slice(t - dt);
	in2_t.data = in2.x.slice(t);
	out_t.data = out.x.slice(t);

	op.forward(in_t, in2_t, out_t, 1.0);
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

	op.backward(in_t, in2_t, out_t, out_err_t, in_err_t, in2_err_t, 1.0);
	//op.backward_weights(in_t, );
}

void VolumeOperation2::forward_dry_run() {
	op.forward_dry_run(in_t, in2_t, out_t);
}
