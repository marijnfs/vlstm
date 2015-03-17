#ifndef __VLSTM_H__
#define __VLSTM_H__

#include <string>
#include <map>
#include <vector>
#include "volume.h"

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
	void clear();
	void init_normal(F mean, F std);
	void update(float lr);

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
	std::vector<Parametrised<F>*> parameters;
 	VolumeSet *vin, *vout;
};

struct VLSTM {
	VLSTM(VolumeShape shape, int kg, int ko, int c);

	void forward();
	void backward();
	void clear();
	void init_normal(F mean, F std);
	void update(float lr);

	VolumeSet x, y;
	Volume6DSet x6, y6;

	std::vector<LSTMOperation*> operations;


};

#endif
