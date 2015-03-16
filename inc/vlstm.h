#ifndef __VLSTM_H__
#define __VLSTM_H__

#include <string>
#include <map>
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

struct LSTMOperation {
	LSTMOperation(VolumeShape in, int kg, int ko, int c);
	LSTMOperation(VolumeSet &in, VolumeSet &out, int kg, int ko, int c);

 	void add_op(std::string in, std::string out, Operation<F> &op, bool delay = false);
 	void add_op(std::string in, std::string in2, std::string out, Operation2<F> &op, bool delay = false);

 	void add_volume(std::string name, VolumeShape shape);
	void add_volume(std::string name, VolumeSet &set);

	bool exists(std::string name);

 	void forward_dry_run();
 	void forward();
 	void backward();

	void add_operations();

	VolumeShape output_shape(VolumeShape in, Operation<F> &op);
	VolumeShape output_shape(VolumeShape in, Operation2<F> &op);

	VolumeSet &input() { return *vin; }
	VolumeSet &output() { return *vout; }

	int T;

	ConvolutionOperation<F> xi, hi; //input gate
	ConvolutionOperation<F> xr, hr; //remember gate (forget gates dont make sense!)
	ConvolutionOperation<F> xs, hs; //cell input
	ConvolutionOperation<F> xo, ho, co; //output gate

	GateOperation<F>		 gate;   //for gating	
	SigmoidOperation<F> sig;
	TanhOperation<F> tan;

	VolumeShape in_shape;
 	std::map<std::string, VolumeSet*> volumes;

 	std::vector<TimeOperation*> operations;

 	VolumeSet *vin, *vout;
};

struct VLSTM {
	VLSTM(VolumeShape shape, int kg, int ko, int c);
	
	void forward();
	void backward();

	std::vector<Parametrised<F>*> params;

	VolumeSet x, y;
	Volume6DSet x6, y6;

	std::vector<LSTMOperation*> operations;


};

#endif
