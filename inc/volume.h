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

std::ostream &operator<<(std::ostream &out, VolumeShape shape);

struct Volume {
	Volume(VolumeShape shape);

	float *slice(int z);
	TensorShape slice_shape();
	void zero();
	void init_normal(F mean, F std);
	void fill(F val);
  	int size();

	VolumeShape shape;
	F *data;
	int slice_size;
};

struct VolumeSet {
	VolumeSet(VolumeShape shape);

	Volume x, diff;
	VolumeShape shape;
};

struct Volume6DSet {
	// order: [x,y,z], [y,x,z], [x, z, y]
	//

	Volume6DSet(VolumeShape shape);

	std::vector<VolumeSet*> volumes;
	std::vector<Volume*> x, diff;
	VolumeShape shape;
};


#endif
