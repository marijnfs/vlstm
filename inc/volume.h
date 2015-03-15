#ifndef __VOLUME_H__
#define __VOLUME_H__

#include <vector>
#include "tensor.h"

typedef F float;

struct VolumeShape {
	int z, c, w, h;

	int size();
};

struct Volume {
	Volume(VolumeShape shape);

	float *slice(int z);
	Tensor<F> create_slice_tensor();
	void zero();
	void init_normal(F mean, F std);
	void fill(F val);
  	int size() const;

	VolumeShape shape;
	F *data;
	int slice_size;
};

struct VolumeSet {
	Volume x, diff;
};

struct VolumeSet3D {
	VolumeSet3D(VolumeShape shape);
	std::vector<VolumeSet> volumes;
};

#endif
