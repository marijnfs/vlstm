#ifndef __VLSTM_H__
#define __VLSTM_H__

#include <vector>
#include "volume.h"


struct VLSTM {
	VolumeSet3D x, rg, ig, og, c, h;
	VolumeSet in, out; //x as single input in case of layering


	std::vector<Parametrised<F>*> params;
	std::vector<Operation<F>*> operations;

	std::vector<Tensor<F>*> in_tensors, out_tensors;
};

#endif
