#ifndef __VLSTM_H__
#define __VLSTM_H__

#include <string>
#include <map>
#include <vector>

#include "volume.h"
#include "operations.h"
#include "volumeoperation.h"



struct LSTMOperation {
	LSTMOperation(VolumeShape in, int kg, int ko, int c, VolumeSetMap *reuse = 0);

	void add_op(std::string in, std::string out, Operation<F> &op, bool delay = false, VolumeSetMap *reuse = 0, float beta = 1.0);
 	void add_op(std::string in, std::string in2, std::string out, Operation2<F> &op, bool delay = false, VolumeSetMap *reuse = 0, float beta = 1.0);

 	void add_volume(std::string name, VolumeShape shape, VolumeSetMap *reuse = 0);
	void add_volume(std::string name, VolumeSet &set);

	bool exists(std::string name);

 	void forward_dry_run();
 	void forward();
 	void backward();
 	void scale_grad();

	void add_operations(VolumeSetMap *reuse = 0);
	void clear();
	void clear_grad();
	void init_normal(F mean, F std);
	void init_uniform(F var);

	void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &grads);
	void share(LSTMOperation &o);

	void update(float lr);

	VolumeSet &input() { return *vin; }
	VolumeSet &output() { return *vout; }

	int T;

	ConvolutionOperation<F> xi, hi; //input gate
	ConvolutionOperation<F> xr, hr; //remember gate (forget gates dont make sense!)
	ConvolutionOperation<F> xs, hs; //cell input
	ConvolutionOperation<F> xo, ho;//, co; //output gate

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
	VLSTMOperation(VolumeShape shape, int kg, int ko, int c, VolumeSetMap &vsm);

	void forward(Volume &in, Volume &out);
	void backward(VolumeSet &in, VolumeSet &out);

	void forward_dry_run(Volume &in, Volume &out);
	void register_params(std::vector<CudaPtr<F>> &params, std::vector<CudaPtr<F>> &grads);
	VolumeShape output_shape(VolumeShape s);

	void sharing();
	void clear();
	void clear_grad();
	void init_normal(F mean, F std);
	void init_uniform(F var);
	void update(float lr);
	void describe(std::ostream &out) { out << "vlstm " << kg << " " << ko << " " << c; }

	int kg, ko;
	int c;

	std::vector<LSTMOperation*> operations;
};

struct UniVLSTMOperation : public VLSTMOperation {
	UniVLSTMOperation(VolumeShape shape, int kg, int ko, int c, VolumeSetMap &vsm);

	void describe(std::ostream &out) { out << "uni vlstm " << kg << " " << ko << " " << c; }
	void forward(Volume &in, Volume &out);
	void backward(VolumeSet &in, VolumeSet &out);
};

#endif
