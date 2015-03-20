#ifndef __VLSTM_H__
#define __VLSTM_H__

#include <string>
#include <map>
#include <vector>

#include "volume.h"
#include "operations.h"
#include "volumeoperation.h"

struct LSTMOperation {
	typedef std::map<std::string, VolumeSet*> VolumeSetMap;

	LSTMOperation(VolumeShape in, int kg, int ko, int c);
	LSTMOperation(VolumeSet &in, VolumeSet &out, int kg, int ko, int c);
	LSTMOperation(VolumeSet &in, VolumeSet &out, VolumeSetMap *reuse, int kg, int ko, int c);

	void add_op(std::string in, std::string out, Operation<F> &op, bool delay = false, VolumeSetMap *reuse = 0);
 	void add_op(std::string in, std::string in2, std::string out, Operation2<F> &op, bool delay = false, VolumeSetMap *reuse = 0);

 	void add_volume(std::string name, VolumeShape shape, VolumeSetMap *reuse = 0);
	void add_volume(std::string name, VolumeSet &set);

	bool exists(std::string name);

 	void forward_dry_run();
 	void forward();
 	void backward();

	void add_operations(VolumeSetMap *reuse = 0);
	void clear();
	void init_normal(F mean, F std);
	void update(float lr);

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
 	VolumeSetMap volumes;

 	std::vector<TimeOperation*> operations;
	std::vector<Parametrised<F>*> parameters;
 	VolumeSet *vin, *vout;
};

struct VLSTMOperation : public VolumeOperation {
	VLSTMOperation(VolumeShape shape, int kg, int ko, int c);

	void forward(Volume &in, Volume &out);
	void backward(Volume &in, Volume &out, Volume &out_grad, Volume &in_grad);
	VolumeShape output_shape(VolumeShape input);
	void forward_dry_run(Volume &in, Volume &out);

	void clear();
	void init_normal(F mean, F std);
	//void update(float lr);

	std::vector<LSTMOperation*> operations;


};

#endif
