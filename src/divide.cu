#include "divide.h"
#include "util.h"
#include "rand.h"

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
		// default:
		// 	throw "";
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

__global__ void copy_subvolume_kernel(VolumeShape inshape, VolumeShape outshape, float *in, float *out,
	VolumeShape in2shape, VolumeShape out2shape,
	float *in2, float *out2, int xs, int ys, int zs, bool xflip, bool yflip, bool zflip) {

	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= outshape.w || y >= outshape.h || z >= outshape.z)
		return;

	int outx = xflip ? (outshape.w - 1 - x) : x;
	int outy = yflip ? (outshape.h - 1 - y) : y;
	int outz = zflip ? (outshape.z - 1 - z) : z;
	int in_index = get_index(inshape.w, inshape.h, inshape.z, inshape.c, x+xs, y+ys, z+zs);
	int in2_index = get_index(in2shape.w, in2shape.h, in2shape.z, in2shape.c, x+xs, y+ys, z+zs);
	int out_index = get_index(outshape.w, outshape.h, outshape.z, outshape.c, outx, outy, outz);
	int out2_index = get_index(out2shape.w, out2shape.h, out2shape.z, out2shape.c, outx, outy, outz);

	copy_c(in + in_index, out + out_index, inshape.w * inshape.h, outshape.w * outshape.h, outshape.c);
	copy_c(in2 + in2_index, out2 + out2_index, in2shape.w * in2shape.h, out2shape.w * out2shape.h, out2shape.c);


}

__global__ void copy_subvolume_test_kernel(VolumeShape inshape, VolumeShape outshape, float *in, float *out, int xs, int ys, int zs) {

	int x(threadIdx.x + blockDim.x * blockIdx.x);
	int y(threadIdx.y + blockDim.y * blockIdx.y);
	int z(threadIdx.z + blockDim.z * blockIdx.z);

	if (x >= outshape.w || y >= outshape.h || z >= outshape.z)
		return;

	int outx = x;
	int outy = y;
	int outz = z;
	int in_index = get_index(inshape.w, inshape.h, inshape.z, inshape.c, x+xs, y+ys, z+zs);
	int out_index = get_index(outshape.w, outshape.h, outshape.z, outshape.c, outx, outy, outz);

	copy_c(in + in_index, out + out_index, inshape.w * inshape.h, outshape.w * outshape.h, outshape.c);
}

void divide(Volume &from, Volume &to, int n) {
	VolumeShape shape = from.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (shape.w + BW - 1) / BW, (shape.h + BH - 1) / BH, shape.z );

	divide_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, from.data(), to.data(), n);
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

	combine_kernel<<<dimGrid, dimBlock>>>(shape.w, shape.h, shape.z, shape.c, to.data(), from.data(), n);
	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}

void copy_subvolume(Volume &in, Volume &out, Volume &in2, Volume &out2, bool xflip, bool yflip, bool zflip) {
	VolumeShape inshape = in.shape;
	VolumeShape outshape = out.shape;
	VolumeShape in2shape = in2.shape;
	VolumeShape out2shape = out2.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (outshape.w + BW - 1) / BW, (outshape.h + BH - 1) / BH, outshape.z );

	int x = Rand::randn(in.shape.w - out.shape.w + 1);
	int y = Rand::randn(in.shape.h - out.shape.h + 1);
	int z = Rand::randn(in.shape.z - out.shape.z + 1);
	// cout <<"copy_subvolume-inshape " << inshape.w << " " << inshape.h << " " << inshape.c << endl;
	// cout <<"copy_subvolume-outshape " << outshape.w << " " << outshape.h << " " << outshape.c << endl;
	cout <<"copy_subvolume-idx " << x << " " << y << " " << z << endl;

	copy_subvolume_kernel<<<dimGrid, dimBlock>>>(inshape, outshape, in.data(), out.data(),
		in2shape, out2shape, in2.data(), out2.data(), x, y, z, xflip, yflip, zflip);


	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}

void copy_subvolume_test(Volume &in, Volume &out, int stx, int sty, int stz) {
	VolumeShape inshape = in.shape;
	VolumeShape outshape = out.shape;

	//primitive blocksize determination
	int const BLOCKSIZE(1024);
	int const BW(32);
	int const BH = BLOCKSIZE / BW;

	dim3 dimBlock( BW, BH, 1 );
	dim3 dimGrid( (outshape.w + BW - 1) / BW, (outshape.h + BH - 1) / BH, outshape.z );

	int x = stx;
	int y = sty;
	int z = stz;
	cout <<"copy_subvolume_test-idx " << x << " " << y << " " << z << endl;

	copy_subvolume_test_kernel<<<dimGrid, dimBlock>>>(inshape, outshape, in.data(), out.data(), x, y, z);

	handle_error( cudaGetLastError() );
	handle_error( cudaDeviceSynchronize());
}