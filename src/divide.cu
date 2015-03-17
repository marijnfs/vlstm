#include "divide.h"
#include "util.h"

using namespace std;

__global__ void divide_kernel(int X, int Y, int Z, int C, float const *in, float *outn, float *outs, float *oute, float *outw, float *outf, float *outb) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= X || y >= Y || z >= Z)
		return;

	int in_index = get_index(X, Y, Z, C, x, y, z);
	copy_c(in + in_index, outn + get_index(X, Y, Z, C, x, y,         z), X * Y, C);
	copy_c(in + in_index, outs + get_index(X, Y, Z, C, x, y, Z - 1 - z), X * Y, C);

	copy_c(in + in_index, oute + get_index(Z, Y, X, C, z, y,         x), Z * Y, C);
	copy_c(in + in_index, outw + get_index(Z, Y, X, C, z, y, X - 1 - x), Z * Y, C);

	copy_c(in + in_index, outf + get_index(X, Z, Y, C, x, z,         y), X * Z, C);
	copy_c(in + in_index, outb + get_index(X, Z, Y, C, x, z, Y - 1 - y), X * Z, C);
}

__global__ void combine_kernel(int X, int Y, int Z, int C, float *in, float const *outn, float const *outs, float const *oute, float const *outw, float const *outf, float const *outb) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= X || y >= Y || z >= Z)
		return;

	int in_index = get_index(X, Y, Z, C, x, y, z);
	add_c(outn + get_index(X, Y, Z, C, x, y,         z), in + in_index, X * Y, C);
	add_c(outs + get_index(X, Y, Z, C, x, y, Z - 1 - z), in + in_index, X * Y, C);

	add_c(oute + get_index(Z, Y, X, C, z, y,         x), in + in_index, Z * Y, C);
	add_c(outw + get_index(Z, Y, X, C, z, y, X - 1 - x), in + in_index, Z * Y, C);

	add_c(outf + get_index(X, Z, Y, C, x, z,         y), in + in_index, X * Z, C);
	add_c(outb + get_index(X, Z, Y, C, x, z, Y - 1 - y), in + in_index, X * Z, C);

}

void divide(Volume &v, vector<Volume*> &to) {
	VolumeShape shape = v.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (shape.w - 1) / BW + 1, (shape.h - 1) / BH + 1, shape.z );

	divide_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, v.data,
		to[0]->data, to[1]->data, to[2]->data,
		to[3]->data, to[4]->data, to[5]->data);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}

void combine(vector<Volume*> &v, Volume &to) {
	VolumeShape shape = to.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (shape.w - 1) / BW + 1, (shape.h - 1) / BH + 1, shape.z );

	combine_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, to.data,
		v[0]->data, v[1]->data, v[2]->data,
		v[3]->data, v[4]->data, v[5]->data);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}
