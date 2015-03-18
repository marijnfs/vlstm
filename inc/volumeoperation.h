#ifndef __VOLUMEOPERATION_H__
#define __VOLUMEOPERATION_H__

#include "operations.h"
#include "volume.h"

struct TimeOperation {
	virtual void forward(int t) = 0;
	virtual void backward(int t) = 0;
	virtual void forward_dry_run() = 0;
};

struct VolumeOperation : public TimeOperation
{
	VolumeOperation(Operation<F> &op, VolumeSet &in, VolumeSet &out, int dt, bool first);

	void forward(int t);
	void backward(int t);
	void forward_dry_run();

	VolumeShape output_shape(VolumeShape input);

	Operation<F> &op;
	int T, dt;
	bool first;

	VolumeSet &in, &out;

	Tensor<F> in_t, out_t;
	Tensor<F> in_err_t, out_err_t;

};

struct VolumeOperation2 : public TimeOperation
{
	VolumeOperation2(Operation2<F> &op, VolumeSet &in, VolumeSet &in2, VolumeSet &out, int dt, bool first);

	void forward(int t);
	void backward(int t);
	void forward_dry_run();

	VolumeShape output_shape(VolumeShape input);

	Operation2<F> &op;
	int T, dt;
	bool first;
	VolumeSet &in, &in2, &out;

	Tensor<F> in_t, in2_t, out_t;
	Tensor<F> in_err_t, in2_err_t, out_err_t;
};


#endif
