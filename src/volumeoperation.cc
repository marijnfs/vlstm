#include "volumeoperation.h"
#include "vlstm.h"

using namespace std;

FCVolumeOperation::FCVolumeOperation(VolumeShape shape, int in_map, int out_map) :
	op(in_map, out_map, 1, 1),
	c(out_map),
	tin(shape.z, in_map, shape.w, shape.h),
	tout(shape.z, out_map, shape.w, shape.h),
	tin_err(shape.z, in_map, shape.w, shape.h),
	tout_err(shape.z, out_map, shape.w, shape.h)
{}

void FCVolumeOperation::forward(Volume &in, Volume &out)  {
	tin.data = in.data;
	tout.data = out.data;
	op.forward(tin, tout);
}

void FCVolumeOperation::backward_weights(VolumeSet &in, VolumeSet &out){
	tin.data = in.x.data;
	tout_err.data = out.diff.data;

	op.backward_weights(tin, tout_err);
}

void FCVolumeOperation::backward(VolumeSet &in, VolumeSet &out) {
	tin.data = in.x.data;
	tout.data = out.x.data;
	tin_err.data = in.diff.data;
	tout_err.data = out.diff.data;

	op.backward(tin, tout, tout_err, tin_err);
}

void FCVolumeOperation::forward_dry_run(Volume &in, Volume &out) {
	tin.data = in.data;
	tout.data = out.data;
	op.forward_dry_run(tin, tout);
}

void FCVolumeOperation::init_normal(float mean, float std) {
	op.init_normal(mean, std);
}

VolumeShape FCVolumeOperation::output_shape(VolumeShape s) {
	return VolumeShape{s.z, c, s.w, s.h};
}


SoftmaxVolumeOperation::SoftmaxVolumeOperation(VolumeShape shape) :
	tin(shape.z, shape.c, shape.w, shape.h),
	tout(shape.z, shape.c, shape.w, shape.h),
	tin_err(shape.z, shape.c, shape.w, shape.h),
	tout_err(shape.z, shape.c, shape.w, shape.h)
{}

void SoftmaxVolumeOperation::forward(Volume &in, Volume &out){
	tin.data = in.data;
	tout.data = out.data;
	op.forward(tin, tout);
}

void SoftmaxVolumeOperation::backward(VolumeSet &in, VolumeSet &out){
	tin.data = in.x.data;
	tout.data = out.x.data;
	tin_err.data = in.diff.data;
	tout_err.data = out.diff.data;

	op.backward(tin, tout, tout_err, tin_err);
}

//Tanh Operation
TanhVolumeOperation::TanhVolumeOperation(VolumeShape shape) :
	tin(shape.z, shape.c, shape.w, shape.h),
	tout(shape.z, shape.c, shape.w, shape.h),
	tin_err(shape.z, shape.c, shape.w, shape.h),
	tout_err(shape.z, shape.c, shape.w, shape.h)
{}

void TanhVolumeOperation::forward(Volume &in, Volume &out){
	tin.data = in.data;
	tout.data = out.data;
	op.forward(tin, tout);
}

void TanhVolumeOperation::backward(VolumeSet &in, VolumeSet &out){
	tin.data = in.x.data;
	tout.data = out.x.data;
	tin_err.data = in.diff.data;
	tout_err.data = out.diff.data;

	op.backward(tin, tout, tout_err, tin_err);
}


//Sigmoid Operation
SigmoidVolumeOperation::SigmoidVolumeOperation(VolumeShape shape) :
	tin(shape.z, shape.c, shape.w, shape.h),
	tout(shape.z, shape.c, shape.w, shape.h),
	tin_err(shape.z, shape.c, shape.w, shape.h),
	tout_err(shape.z, shape.c, shape.w, shape.h)
{}

void SigmoidVolumeOperation::forward(Volume &in, Volume &out){
	tin.data = in.data;
	tout.data = out.data;
	op.forward(tin, tout);
}

void SigmoidVolumeOperation::backward(VolumeSet &in, VolumeSet &out){
	tin.data = in.x.data;
	tout.data = out.x.data;
	tin_err.data = in.diff.data;
	tout_err.data = out.diff.data;

	op.backward(tin, tout, tout_err, tin_err);
}
//Volume Operation
TimeOperation1::TimeOperation1(Operation<F> &op_, VolumeSet &in_, VolumeSet &out_, int dt_, bool first_) :
	op(op_), T(in_.x.shape.z), dt(dt_), first(first_),
	in(in_), out(out_),
	in_t(in_.x.slice_shape(), 0),
	out_t(out_.x.slice_shape(), 0),
	in_err_t(in_.x.slice_shape(), 0),
	out_err_t(out_.x.slice_shape(), 0)
{}

void TimeOperation1::forward(int t) {
	if (dt > t)
		return;
	in_t.data = in.x.slice(t - dt);
	out_t.data = out.x.slice(t);

	op.forward(in_t, out_t, 1.0);
}

void TimeOperation1::backward(int t) {
	if (dt > t)
		return;
	in_t.data = in.x.slice(t - dt);
	out_t.data = out.x.slice(t);

	in_err_t.data = in.diff.slice(t - dt);
	out_err_t.data = out.diff.slice(t);

	op.backward(in_t, out_t, out_err_t, in_err_t, 1.0);
	op.backward_weights(in_t, out_err_t, 1.0);
}

void TimeOperation1::forward_dry_run() {
	cout << in_t.shape() << " " << out_t.shape() << endl;
	op.forward_dry_run(in_t, out_t);
}

TimeOperation2::TimeOperation2(Operation2<F> &op_, VolumeSet &in_, VolumeSet &in2_, VolumeSet &out_, int dt_, bool first_) :
	op(op_),T(in_.x.shape.z), dt(dt_), in(in_), first(first_), in2(in2_), out(out_),
	in_t(in_.x.slice_shape(), 0),
	in2_t(in2_.x.slice_shape(), 0),
	out_t(out_.x.slice_shape(), 0),
	in_err_t(in_.x.slice_shape(), 0),
	in2_err_t(in2_.x.slice_shape(), 0),
	out_err_t(out_.x.slice_shape(), 0)
{}

void TimeOperation2::forward(int t) {
	if (dt > t)
		return;
	//we only delay the first input
	in_t.data = in.x.slice(t - dt);
	in2_t.data = in2.x.slice(t);
	out_t.data = out.x.slice(t);

	op.forward(in_t, in2_t, out_t, 1.0);
}

void TimeOperation2::backward(int t) {
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

void TimeOperation2::forward_dry_run() {
	op.forward_dry_run(in_t, in2_t, out_t);
}

VolumeShape output_shape(VolumeShape in, Operation<F> &op) {
	TensorShape out = op.output_shape(TensorShape{1, in.c, in.w, in.h});
	return VolumeShape{in.z, out.c, out.w, out.h};
}

VolumeShape output_shape(VolumeShape in, Operation2<F> &op) {
	TensorShape out = op.output_shape(TensorShape{1, in.c, in.w, in.h});
	return VolumeShape{in.z, out.c, out.w, out.h};
}