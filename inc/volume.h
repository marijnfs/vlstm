#ifndef __VOLUME_H__
#define __VOLUME_H__

#include <vector>
#include "tensor.h"

//initially just float
typedef float F;

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

struct Volume6D {
	// order: [x,y,z], [y,x,z], [x, z, y]
	//

	Volume6D(VolumeShape shape);

	std::vector<Volume*> volumes;
	VolumeShape baseshape;
};

struct Volume6DSet {
	Volume6D x, diff;
};

#endif
