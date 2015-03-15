#ifndef __VLSTM_H__
#define __VLSTM_H__

#include <vector>
#include "volume.h"


struct LSTMForward {

	ConvolutionOperation xi, hi; //input gate
	ConvolutionOperation xr, hr; //remember gate (forget gates dont make sense!)
	ConvolutionOperation xs, hs; //cell input
	GateOperation		 gate;   //for gating
	ConvolutionOperation xo, ho, co; //output gate
	
	SigmoidOperation fi, fr, fo;
	TanhOperation fs;
	
	std::vector<Tensor<F>*> in_tensors, out_tensors;
};

struct VLSTM {
	VolumeSet3D x, rg, ig, og, c, h;
	VolumeSet in, out; //x as single input in case of layering


	std::vector<Parametrised<F>*> params;
	std::vector<Operation<F>*> operations;

	
};

#endif
