#include "volume.h"
#include "util.h"
#include "img.h"
#include <cmath>

using namespace std;

Volume::Volume(VolumeShape shape_) : shape(shape_), slice_size(shape_.c * shape_.w * shape_.h) {
	size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	cout << "allocating volume: " << shape << " nfloats: " << even_size << endl;
	handle_error( cudaMalloc((void**)&data, sizeof(float) * even_size) 	);
	zero();
}

Volume::Volume(VolumeShape shape_, Volume &reuse_buffer) : shape(shape_), slice_size(shape_.c * shape_.w * shape_.h) {
	data = reuse_buffer.data;
	zero();
}


float *Volume::slice(int z) {
	return data + z * slice_size;
}

TensorShape Volume::slice_shape() {
	return TensorShape{1, shape.c, shape.h, shape.w};
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

void Volume::from_volume(Volume &other) {
	if (size() != other.size()) {
 			throw StringException("sizes don't match");
	}
	handle_error( cudaMemcpy(data, other.data, other.size() * sizeof(F), cudaMemcpyDeviceToDevice));
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
	return sqrt(result) / size();
}

std::vector<F> Volume::to_vector() {
	vector<F> vec(size());
	handle_error( cudaMemcpy(&vec[0], data, vec.size() * sizeof(F), cudaMemcpyDeviceToHost));
	return vec;
}

void Volume::draw_slice(string filename, int slice) {
	vector<F> data = to_vector();

	write_img1c(filename, shape.w, shape.h, &data[slice * slice_size]);
}

int VolumeShape::size() {
	return z * c * w * h;
}


std::ostream &operator<<(std::ostream &out, VolumeShape shape) {
	return out << "[z:" << shape.z << " c:" << shape.c << " w:" << shape.w << " h:" << shape.h << "]";
}

VolumeSet::VolumeSet(VolumeShape shape_) : x(shape_), diff(shape_), shape(shape_)
{}

VolumeSet::VolumeSet(VolumeShape shape_, VolumeSet &reuse_buffer) : x(shape_, reuse_buffer.x), diff(shape_, reuse_buffer.diff), shape(shape_)
{}

void VolumeSet::zero() {
	x.zero();
	diff.zero();
}

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

void Volume6DSet::zero() {
	for (auto &v : volumes)
		v->zero();
}
