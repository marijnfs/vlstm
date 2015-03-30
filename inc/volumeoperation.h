#ifndef __VOLUMEOPERATION_H__
#define __VOLUMEOPERATION_H__

#include "operations.h"
#include "volume.h"

struct VolumeOperation {
	virtual void forward(Volume &in, Volume &out){}

	virtual void backward_weights(VolumeSet &in, VolumeSet &out){}
	virtual void backward(VolumeSet &in, VolumeSet &out){}
	virtual VolumeShape output_shape(VolumeShape input) { return input; }
	virtual void update(float lr) {}
	virtual void forward_dry_run(Volume &in, Volume &out) {}
	virtual void init_normal(float mean, float std) {}
	virtual void init_uniform(float std) {}
	virtual void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &grads) {}
};

struct FCVolumeOperation : public VolumeOperation {
	FCVolumeOperation(VolumeShape shape, int in_map, int out_map);

	void forward(Volume &in, Volume &out);
	void backward_weights(VolumeSet &in, VolumeSet &out);
	void backward(VolumeSet &in, VolumeSet &out);
	void forward_dry_run(Volume &in, Volume &out);
	void update(float lr);

	void init_normal(float mean, float std);
	void init_uniform(float std);
	void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &grads);
	VolumeShape output_shape(VolumeShape s);

	ConvolutionOperation<F> op;
	VolumeShape shape;
	int c;
	Tensor<F> tin, tout;
	Tensor<F> tin_err, tout_err;
};

struct SoftmaxVolumeOperation : public VolumeOperation {
	SoftmaxVolumeOperation(VolumeShape shape);

	void forward(Volume &in, Volume &out);
	void backward(VolumeSet &in, VolumeSet &out);

	SoftmaxOperation<F> op;
	Tensor<F> tin, tout;
	Tensor<F> tin_err, tout_err;
};

struct TanhVolumeOperation : public VolumeOperation {
	TanhVolumeOperation(VolumeShape shape);

	void forward(Volume &in, Volume &out);
	void backward(VolumeSet &in, VolumeSet &out);

	TanhOperation<F> op;
	Tensor<F> tin, tout;
	Tensor<F> tin_err, tout_err;
};

struct SigmoidVolumeOperation : public VolumeOperation {
	SigmoidVolumeOperation(VolumeShape shape);

	void forward(Volume &in, Volume &out);
	void backward(VolumeSet &in, VolumeSet &out);

	SigmoidOperation<F> op;
	Tensor<F> tin, tout;
	Tensor<F> tin_err, tout_err;
};

struct TimeOperation {
	virtual void forward(int t) = 0;
	virtual void backward(int t) = 0;
	virtual void forward_dry_run() = 0;
};

struct TimeOperation1 : public TimeOperation
{
	TimeOperation1(Operation<F> &op, VolumeSet &in, VolumeSet &out, int dt, float beta = 1.0);

	void forward(int t);
	void backward(int t);
	void forward_dry_run();

	VolumeShape output_shape(VolumeShape input);

	Operation<F> &op;
	int T, dt;
	float beta;

	VolumeSet &in, &out;

	Tensor<F> in_t, out_t;
	Tensor<F> in_err_t, out_err_t;

};

struct TimeOperation2 : public TimeOperation
{
	TimeOperation2(Operation2<F> &op, VolumeSet &in, VolumeSet &in2, VolumeSet &out, int dt, float beta = 1.0);

	void forward(int t);
	void backward(int t);
	void forward_dry_run();

	// VolumeShape output_shape(VolumeShape input);

	Operation2<F> &op;
	int T, dt;
	float beta;
	VolumeSet &in, &in2, &out;

	Tensor<F> in_t, in2_t, out_t;
	Tensor<F> in_err_t, in2_err_t, out_err_t;
};



VolumeShape output_shape(VolumeShape in, Operation<F> &op);
VolumeShape output_shape(VolumeShape in, Operation2<F> &op);


#endif
