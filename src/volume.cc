#include "volume.h"
#include "util.h"
#include <cmath>

using namespace std;

Volume::Volume(VolumeShape shape_) : shape(shape_), slice_size(shape_.c * shape_.w * shape_.h) {
	//handle_error( cudnnCreateTensorDescriptor(&td));
	//handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
	size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	cout << "allocating volume: " << shape << " nfloats: " << even_size << endl;
	handle_error( cudaMalloc((void**)&data, sizeof(float) * even_size) 	);
	zero();
}

float *Volume::slice(int z) {
	return data + z * slice_size;
}

TensorShape Volume::slice_shape() { 
	return TensorShape{1, shape.c, shape.w, shape.h};
}

void Volume::zero() {
	handle_error( cudaMemset(data, 0, sizeof(F) * size()) );
}

void Volume::init_normal(F mean, F std) {
	size_t even_size(((size() + 1) / 2) * 2);
	handle_error( curandGenerateNormal(Handler::curand(), data, even_size, mean, std) );
}

void Volume::fill(F val) {
	throw StringException("not implemented");
}

int Volume::size() {
	return shape.size();
}

Volume &operator-=(Volume &in, Volume &other) {
	assert(in.size() == other.size());
	add_cuda<F>(other.data, in.data, in.size(), -1);
	return in;
}

float Volume::norm() {
	float result(0);	
	handle_error( cublasSdot(Handler::cublas(), size(), data, 1, data, 1, &result) );
	return sqrt(result);
}


int VolumeShape::size() {
	return z * c * w * h;
}

std::ostream &operator<<(std::ostream &out, VolumeShape shape) {
	return out << "[z:" << shape.z << " c:" << shape.c << " w:" << shape.w << " h:" << shape.h << "]";
}

VolumeSet::VolumeSet(VolumeShape shape_) : x(shape_), diff(shape_), shape(shape_) 
{}

Volume6DSet::Volume6DSet(VolumeShape shape_) : shape(shape_) {
	VolumeShape &s(shape);

	volumes.push_back(new VolumeSet(VolumeShape{s.z, s.c, s.w, s.h}));
	volumes.push_back(new VolumeSet(VolumeShape{s.z, s.c, s.w, s.h}));

	volumes.push_back(new VolumeSet(VolumeShape{s.w, s.c, s.z, s.h}));
	volumes.push_back(new VolumeSet(VolumeShape{s.w, s.c, s.z, s.h}));

	volumes.push_back(new VolumeSet(VolumeShape{s.h, s.c, s.w, s.z}));
	volumes.push_back(new VolumeSet(VolumeShape{s.h, s.c, s.w, s.z}));

	for (auto &v : volumes) {
		x.push_back(&(v->x));
		diff.push_back(&(v->diff));
	}
}
