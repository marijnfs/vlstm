#ifndef __DIVIDE_H__
#define __DIVIDE_H__

#include <cuda.h>

__device__ __forceinline__ int get_index(int X, int Y, int Z, int C, int x, int y, int z) {
	return z * (C * X * Y) + x * Y + y;
}

__device__ __forceinline__ void copy_c(float *in, float *out, int slicesize, int C) {
	for (size_t c(0); c < C; ++c)
		out[c * slicesize] = in[c * slicesize];
}

__device__ __forceinline__ void add_c(float *in, float *out, int slicesize, int C) {
	for (size_t c(0); c < C; ++c)
		out[c * slicesize] += in[c * slicesize];
}

#include "divide.h"


__global__ void divide_kernel(int X, int Y, int Z, int C, float const *in, 
	float *outn, float *outs, float *oute, 
	float *outw, float *outf, float *outb);

__global__ void combine_kernel(int X, int Y, int Z, int C, float const *in, 
	float *outn, float *outs, float *oute, 
	float *outw, float *outf, float *outb);

void divide(Volume &v, Volume6D &to);
void combine(Volume6D &v, Volume &to);

#endif
