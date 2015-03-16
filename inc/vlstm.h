#ifndef __VLSTM_H__
#define __VLSTM_H__

#include <vector>
#include "volume.h"

#include "operations.h"

struct TimeOperation {
	virtual void forward(int t) = 0;
	virtual void backward(int t) = 0;
	virtual void forward_dry_run() = 0;
};

struct VolumeOperation : public TimeOperation
{
	VolumeOperation(Operation &op, VolumeSet &in, &out, int dt, bool first);

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
	VolumeOperation2(Operation &op, VolumeSet &in, VolumeSet &in2, VolumeSet &out, int dt, bool first);

	void forward(int t);
	void backward(int t);
	void forward_dry_run();

	VolumeShape output_shape(VolumeShape input);

	Operation<F> &op;
	int T, dt;
	bool first;
	VolumeSet &in, &in2, &out;

	Tensor<F> in_t, in2_t, out_t;
	Tensor<F> in_err_t, in2_err_t, out_err_t;
};

struct LSTMOperation {
	LSTMOperation(VolumeShape in, int kg, int ko, int c);
	LSTMOperation(VolumeSet &in, VolumeSet &out, int kg, int ko, int c);

 	void add_op(string in, string out, Operation &op, bool delay);
 	void add_op(string in, string in2, string out, Operation2 &op, bool delay);

 	void add_volume(std::string name, VolumeShape shape);
	void add_volume(std::string name, VolumeSet &set);

	bool exists(std::string name);

 	void forward_dry_run();
 	void forward();
 	void backward();

	void add_operations();

	VolumeShape output_shape(VolumeShape in, Operation &op);
	VolumeSet &input() { return *vin; }
	VolumeSet &output() { return *vout; }

	int T;

	ConvolutionOperation xi, hi; //input gate
	ConvolutionOperation xr, hr; //remember gate (forget gates dont make sense!)
	ConvolutionOperation xs, hs; //cell input
	ConvolutionOperation xo, ho, co; //output gate

	GateOperation		 gate;   //for gating	
	SigmoidOperation sig;
	TanhOperation tan;

	VolumeShape in_shape;
 	std::map<string, VolumeSet> volumes;

 	std::vector<TimeOperation*> operations;

 	VolumeSet *vin, *vout;
};

struct VLSTM {
	VLSTM(VolumeShape shape, int kg, int ko, int c);
	
	void forward();
	void backward();

	std::vector<Parametrised<F>*> params;
	std::vector<Operation<F>*> operations;


	VolumeSet x, y;
	Volume6DSet x6, y6;

	std::vector<LSTMOperation> operations;


};

#endif
