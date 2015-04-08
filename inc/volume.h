#ifndef __VOLUME_H__
#define __VOLUME_H__

#include <vector>
#include <map>
#include <string>
#include <cassert>
#include "tensor.h"
#include "cudavec.h"

//initially just float
typedef float F;

struct VolumeShape {
	int z, c, w, h;

	int size();
};

std::ostream &operator<<(std::ostream &out, VolumeShape shape);

struct Volume {
	Volume(VolumeShape shape);
	Volume(VolumeShape shape, Volume &reuse_buffer);

	float *slice(int z);
	TensorShape slice_shape();
	void zero();
	void init_normal(F mean, F std);
	void fill(F val);
  	int size();
  	float norm();
  	float norm2();

  	void from_volume(Volume &other);
  	std::vector<F> to_vector();
	void draw_slice(std::string filename, int slice);
	float *data();

	VolumeShape shape;
	CudaVec *buf;
	int slice_size;
};

Volume &operator-=(Volume &in, Volume &other);

struct VolumeSet {
	VolumeSet(VolumeShape shape);
	VolumeSet(VolumeShape shape, VolumeSet &reuse_buffer);
	void zero();

	Volume x, diff;	// x: activation
	VolumeShape shape;
};

typedef std::map<std::string, VolumeSet*> VolumeSetMap;

// struct Volume6DSet {
// 	// order: [x,y,z], [y,x,z], [x, z, y]
// 	//

// 	Volume6DSet(VolumeShape shape);
// 	void zero();

// 	std::vector<VolumeSet*> volumes;
// 	std::vector<Volume*> x, diff;
// 	VolumeShape shape;
// };


#endif
