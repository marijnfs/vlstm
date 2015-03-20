#ifndef __DIVIDE_H__
#define __DIVIDE_H__

#include <cuda.h>
#include "divide.h"
#include "volume.h"

__device__ __forceinline__ int get_index(int X, int Y, int Z, int C, int x, int y, int z) {
	return z * (C * X * Y) + x * Y + y; //CWH, as cudnn
}

__device__ __forceinline__ void copy_c(float const *in, float *out, int slicesizein, int slicesizeout, int C) {
	for (size_t c(0); c < C; ++c)
		out[c * slicesizeout] = in[c * slicesizein];
}

__device__ __forceinline__ void add_c(float const *in, float *out, int slicesizein, int slicesizeout, int C) {
	for (size_t c(0); c < C; ++c)
		out[c * slicesizeout] += in[c * slicesizein];
}

__global__ void divide_kernel(int X, int Y, int Z, int C, float const *in, float *out, int n);

__global__ void combine_kernel(int X, int Y, int Z, int C, float *in, float const *out, int n);

void divide(Volume &v, Volume &to, int n);
void combine(Volume &v, Volume &to, int n);

#endif
