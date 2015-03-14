#ifndef __VOLUME_H__
#define __VOLUME_H__

#include <vector>
#include "tensor.h"

struct VolumeShape {
	int z, c, w, h;
};

struct Volume {
	Volume(VolumeShape shape);

	float *slice(int z);


	int z, c, w, h;
	F *data;

	Tensor create_slice_tensor();
	void zero();
	void init_normal(F mean, F std);
	void fill(F val);
};

struct VolumeSet {
	Volume x, diff;
};

struct VolumeSet3D {
	VolumeSet3D(VolumeShape shape);
	std::vector<VolumeSet> volumes;
};

#endif
