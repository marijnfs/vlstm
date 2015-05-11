#include "volume.h"
#include "util.h"
#include "img.h"
#include <cmath>

using namespace std;

Volume::Volume(VolumeShape shape_) : shape(shape_), slice_size(shape_.c * shape_.w * shape_.h) {
	// size_t even_size(((size() + 1) / 2) * 2); //we want multiple of two for curand
	cout << "allocating volume: " << shape << " nfloats: " << size() << endl;
	//handle_error( cudaMalloc((void**)&data, sizeof(float) * size()) 	);
	buf = new CudaVec(size());
	zero();
}

Volume::Volume(VolumeShape shape_, Volume &reuse_buffer) : shape(shape_), slice_size(shape_.c * shape_.w * shape_.h) {
	if (size() > reuse_buffer.buf->n) {
		cout << "resizing " << size() << " / "  << reuse_buffer.buf->n << endl;
		reuse_buffer.buf->resize(size());
	}
	//data = reuse_buffer.data;
	buf = reuse_buffer.buf;
	zero();
}


float *Volume::slice(int z) {
	return buf->data + z * slice_size;
}

TensorShape Volume::slice_shape() {
	return TensorShape{1, shape.c, shape.h, shape.w};
}

void Volume::zero() {
	buf->zero();
}

void Volume::rand_zero(float p) {
	buf->rand_zero(p);
}


void Volume::init_normal(F mean, F std) {
	// size_t even_size(((size() + 1) / 2) * 2);
	// zero();
	buf->init_normal(mean, std);
	// ::init_normal(data, size(), mean, std);
	// handle_error( curandGenerateNormal(Handler::curand(), data, even_size, mean, std) );
}

void Volume::fill(F val) {
	throw StringException("not implemented");
}

void Volume::from_volume(Volume &other) {
	if (size() != other.size()) {
 			throw StringException("sizes don't match");
	}
	handle_error( cudaMemcpy(buf->data, other.buf->data, other.size() * sizeof(F), cudaMemcpyDeviceToDevice));
}

int Volume::size() {
	return shape.size();
}

Volume &operator-=(Volume &in, Volume &other) {
	assert(in.size() == other.size());
	add_cuda<F>(other.buf->data, in.buf->data, in.size(), -1);
	return in;
}

float Volume::norm() {
	float result(0);
	handle_error( cublasSdot(Handler::cublas(), size(), buf->data, 1, buf->data, 1, &result) );
	return sqrt(result) / size();
}

float Volume::norm2() {
	float result(0);
	handle_error( cublasSdot(Handler::cublas(), size(), buf->data, 1, buf->data, 1, &result) );
	return result;
}

std::vector<F> Volume::to_vector() {
	vector<F> vec(size());
	// cout << size() << " " << data << " ";
	handle_error( cudaMemcpy(&vec[0], buf->data, vec.size() * sizeof(F), cudaMemcpyDeviceToHost));
	return vec;
}

void Volume::from_vector(std::vector<F> &vec) {
	buf->from_vector(vec);
}

void Volume::thresholding(std::vector<F> &data, float threshold) {
	for (auto &v : data)
		if (v > threshold)	v = 1.0;
		else				v = 0.0;
}

void Volume::draw_slice(string filename, int slice, float th) {
	vector<F> data = to_vector();
	if (th > 0.0)	thresholding(data, th);

	write_img1c(filename, shape.w, shape.h, &data[slice * slice_size]);
}

void Volume::draw_slice_rgb(string filename, int slice) {
	vector<F> data = to_vector();
	vector<F> rgb(shape.w*shape.h*3);

	for(int c(0); c < 3; c++)
		for(int i(0); i < shape.w*shape.h; i++){
			rgb[i*3+c] = data[slice * slice_size + c * shape.w*shape.h + i];
		}
	write_img(filename, 3, shape.w, shape.h, &rgb[0]);
}

void Volume::dropout(float p) {
	buf->rand_zero(p);
}

float *Volume::data() {
	return buf->data;
}


int VolumeShape::size() {
	return z * c * w * h;
}

int VolumeShape::offset(int zz, int cc, int x, int y) {
	return zz * c * w * h + cc * w * h + x * h + y;
}

int VolumeShape::offsetrm(int zz, int cc, int x, int y) {
	return zz * c * w * h + cc * w * h + y * w + x;
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

// Volume6DSet::Volume6DSet(VolumeShape shape_) : shape(shape_) {
// 	VolumeShape &s(shape);

// 	volumes.push_back(new VolumeSet(VolumeShape{s.z, s.c, s.w, s.h}));
// 	volumes.push_back(new VolumeSet(VolumeShape{s.z, s.c, s.w, s.h}));

// 	volumes.push_back(new VolumeSet(VolumeShape{s.w, s.c, s.z, s.h}));
// 	volumes.push_back(new VolumeSet(VolumeShape{s.w, s.c, s.z, s.h}));

// 	volumes.push_back(new VolumeSet(VolumeShape{s.h, s.c, s.w, s.z}));
// 	volumes.push_back(new VolumeSet(VolumeShape{s.h, s.c, s.w, s.z}));

// 	for (auto &v : volumes) {
// 		x.push_back(&(v->x));
// 		diff.push_back(&(v->diff));
// 	}
// }

// void Volume6DSet::zero() {
// 	for (auto &v : volumes)
// 		v->zero();
// }
