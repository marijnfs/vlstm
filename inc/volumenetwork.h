#ifndef __VOLUMENETWORK_H__
#define __VOLUMENETWORK_H__

#include "volume.h"
#include "volumeoperation.h"
#include "cudavec.h"

struct VolumeNetwork {
	VolumeNetwork(VolumeShape shape);

	void forward();
	void backward();
	void forward_dry_run();

	void finish();

	void register_params();

	void set_input(Volume &in);
	float calculate_loss(Volume &target);
	void update(float lr);
	void clear();
	void init_normal(float mean, float std);
	void init_uniform(float std);
	void align_params();
	void position_params(float *pos_param, float *pos_fast_param, float *pos_grad);

	Volume &input();
	Volume &output();

	VolumeShape output_shape() {return last(shapes);}

	void add_vlstm(int kg, int ko, int c);
	void add_univlstm(int kg, int ko, int c);
	void add_fc(int c, float dropout=0.0);
	void add_softmax();
	void add_tanh();
	void add_sigmoid();

	void save(std::string path);
	void load(std::string path);
	void describe(std::ostream &out);

	void set_fast_weights(Tensor<float> &weights);


	std::vector<CudaPtr<F>> params, fast_params;
	std::vector<CudaPtr<F>> grads;

	CudaVec param, fast_param, grad;
	CudaVec a, b, c, d, e, rmse;
	int n_params, n_fast_params;


	std::vector<VolumeOperation*> operations;
	std::vector<VolumeSet*> volumes;
	std::vector<VolumeShape> shapes;

	VolumeSetMap vsm;
};

#endif
