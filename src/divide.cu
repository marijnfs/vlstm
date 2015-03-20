#include "divide.h"
#include "util.h"

using namespace std;

__global__ void divide_kernel(int X, int Y, int Z, int C, float const *in, float *out, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= X || y >= Y || z >= Z)
		return;

	int in_index = get_index(X, Y, Z, C, x, y, z);

	switch(n){
		case 0:
			copy_c(in + in_index, out + get_index(X, Y, Z, C, x, y,         z), X * Y, X * Y, C);
			break;
		case 1:
			copy_c(in + in_index, out + get_index(X, Y, Z, C, x, y, Z - 1 - z), X * Y, X * Y, C);
			break;
		case 2:
			copy_c(in + in_index, out + get_index(Z, Y, X, C, z, y,         x), X * Y, Z * Y, C);
			break;
		case 3:
			copy_c(in + in_index, out + get_index(Z, Y, X, C, z, y, X - 1 - x), X * Y, Z * Y, C);
			break;
		case 4:
			copy_c(in + in_index, out + get_index(X, Z, Y, C, x, z,         y), X * Y, X * Z, C);
			break;
		case 5:
			copy_c(in + in_index, out + get_index(X, Z, Y, C, x, z, Y - 1 - y), X * Y, X * Z, C);
			break;
	}
}

__global__ void combine_kernel(int X, int Y, int Z, int C, float *in, float const *out, int n) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= X || y >= Y || z >= Z)
		return;

	int in_index = get_index(X, Y, Z, C, x, y, z);

	switch(n){
		case 0:
			add_c(out + get_index(X, Y, Z, C, x, y,         z), in + in_index, X * Y, X * Y, C);
			break;
		case 1:
			add_c(out + get_index(X, Y, Z, C, x, y, Z - 1 - z), in + in_index, X * Y, X * Y, C);
			break;
		case 2:
			add_c(out + get_index(Z, Y, X, C, z, y,         x), in + in_index, Z * Y, X * Y, C);
			break;
		case 3:
			add_c(out + get_index(Z, Y, X, C, z, y, X - 1 - x), in + in_index, Z * Y, X * Y, C);
			break;
		case 4:
			add_c(out + get_index(X, Z, Y, C, x, z,         y), in + in_index, X * Z, X * Y, C);
			break;
		case 5:
			add_c(out + get_index(X, Z, Y, C, x, z, Y - 1 - y), in + in_index, X * Z, X * Y, C);
			break;
	}

}

void divide(Volume &from, Volume &to, int n) {
	VolumeShape shape = from.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (shape.w + BW - 1) / BW, (shape.h + BH - 1) / BH, shape.z );

	divide_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, from.data, to.data, n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}

void combine(Volume &from, Volume &to, int n) {
	VolumeShape shape = to.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (shape.w + BW - 1) / BW, (shape.h + BH - 1) / BH, shape.z );

	combine_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, to.data, from.data, n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}
