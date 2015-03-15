#include "divide.h"


__global__ void divide_kernel(int X, int Y, int Z, int C, float const *in, float *outn, float *outs, float *oute, float *outw, float *outf, float *outb) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= X || y >= Y || z >= Z)
		return;
	
	copy_c(in + get_index(X, Y, Z, C, x, y, z), out + get_index(X, Y, Z, C, x, y, z), X, Y, C);
	copy_c(in + get_index(X, Y, Z, C, x, y, z), out + get_index(X, Y, Z, C, x, Y - y, z), X, Y, C);
	copy_c(in + get_index(X, Y, Z, C, x, y, z), out + get_index(Y, X, Z, C, y, x, z), X, Y, C);
	copy_c(in + get_index(X, Y, Z, C, x, y, z), out + get_index(Y, X, Z, C, y, X - x, z), X, Y, C);
	copy_c(in + get_index(X, Y, Z, C, x, y, z), out + get_index(X, Z, Y, C, x, z, y), X, Y, C);
	copy_c(in + get_index(X, Y, Z, C, x, y, z), out + get_index(X, Z, Y, C, x, Z - z, y), X, Y, C);
}

__global__ void combine_kernel(int X, int Y, int Z, int C, float const *in, float *outn, float *outs, float *oute, float *outw, float *outf, float *outb) {
	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= X || y >= Y || z >= Z)
		return;
	
	add_c(out + get_index(X, Y, Z, C, x,     y, z), X, Y, C, in + get_index(X, Y, Z, C, x, y, z));
	add_c(out + get_index(X, Y, Z, C, x, Y - y, z), X, Y, C, in + get_index(X, Y, Z, C, x, y, z));
	add_c(out + get_index(Y, X, Z, C, y,     x, z), X, Y, C, in + get_index(X, Y, Z, C, x, y, z));
	add_c(out + get_index(Y, X, Z, C, y, X - x, z), X, Y, C, in + get_index(X, Y, Z, C, x, y, z));
	add_c(out + get_index(X, Z, Y, C, x,     z, y), X, Y, C, in + get_index(X, Y, Z, C, x, y, z));
	add_c(out + get_index(X, Z, Y, C, x, Z - z, y), X, Y, C, in + get_index(X, Y, Z, C, x, y, z));
}

void divide(Volume &v, Volume6D &to) {
	VolumeShape shape = v.shape;

	dim3 dimBlock( shape.x, shape.y, shape.z );
	dim3 dimGrid( 1 );
	
	divide_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, v.data, 
		to.volumes[1].data, to.volumes[2].data, to.volumes[3].data, 
		to.volumes[4].data, to.volumes[5].data, to.volumes[6].data);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}

void combine(Volume6D &v, Volume &to) {
	VolumeShape shape = to.shape;

	dim3 dimBlock( shape.x, shape.y, shape.z );
	dim3 dimGrid( 1 );
	
	divide_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, to.data, 
		v.volumes[1].data, v.volumes[2].data, v.volumes[3].data, 
		v.volumes[4].data, v.volumes[5].data, v.volumes[6].data);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}
