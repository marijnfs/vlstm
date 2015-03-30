#ifndef __CUDAVEC_H__
#define __CUDAVEC_H__

#include <vector>
#include <cuda.h>
#include "util.h"

struct CudaVec {
	float *data;
	int n;

	CudaVec() : data(0), n(0) { }
	CudaVec(int n_) : data(0), n(0) { resize(n_); }
	~CudaVec() {if (n) cudaFree(data);}
	void resize(int n2) {
		if (n) {
			std::cout << "freeing " << n << std::endl;
			cudaFree(data);
		}
		std::cout <<"allocating " << n2 << std::endl;
		handle_error( cudaMalloc( (void**)&data, sizeof(float) * n2));
		n = n2;
		zero();
	}

	CudaVec &operator=(CudaVec &other) {
		if (n != other.n)
			resize(other.n);
		handle_error( cudaMemcpy(data, other.data, n * sizeof(float), cudaMemcpyDeviceToDevice));
		return *this;
	}

	void zero() {
		handle_error( cudaMemset(data, 0, sizeof(float) * n) );
	}

	void init_normal(float mean, float std) {
		::init_normal<float>(data, n, mean, std);
	}

	std::vector<float> to_vector() {
		std::vector<float> vec(n);
		handle_error( cudaMemcpy(&vec[0], data, n * sizeof(float), cudaMemcpyDeviceToHost));
		return vec;
	}

	void from_vector(std::vector<float> &vec) {
		if (vec.size() != n)
			resize(vec.size());
		handle_error( cudaMemcpy(data, &vec[0], n * sizeof(float), cudaMemcpyHostToDevice));
	}

	CudaVec &sqrt();
	CudaVec &abs();
	CudaVec &clip(float limit);

	CudaVec &operator-=(CudaVec &other);
	CudaVec &operator+=(CudaVec &other);
	CudaVec &operator*=(CudaVec &other);
	CudaVec &operator/=(CudaVec &other);


	CudaVec &operator*=(float v);
	CudaVec &operator+=(float v);

};

__global__ void sqrt_kernel(float *v, int n);
__global__ void abs_kernel(float *v, int n);
__global__ void clip_kernel(float *v, int n, float limit);

__global__ void times_kernel(float *v, float *other, int n);
__global__ void divide_kernel(float *v, float *other, int n);
__global__ void times_scalarf(float *v, float other, int n);
__global__ void add_scalarf(float *v, float other, int n);



#endif
