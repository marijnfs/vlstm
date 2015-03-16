#include "volume.h"
#include "util.h"

Volume::Volume(VolumeShape shape_) : shape(shape_), slice_size(shape_.c * shape_.w * shape_.h) {
	//handle_error( cudnnCreateTensorDescriptor(&td));
	//handle_error( cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)); //CUDNN_TENSOR_NHWC not supported for some reason
	size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	handle_error( cudaMalloc((void**)&data, sizeof(float) * even_size) 	);
	zero();
}

float *Volume::slice(int z) {
	return data + z * slice_size;
}

Tensor<F> Volume::create_slice_tensor() { 
	return Tensor<F>(TensorShape{1, shape.c, shape.w, shape.h}, 0); //0 pointer to prevent allocation
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

int VolumeShape::size() {
	return z * c * w * h;
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
