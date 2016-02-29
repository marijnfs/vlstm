#include "trainer.h"
#include "log.h"

Trainer::Trainer(int param_n, float start_lr_, float end_lr_, float half_time) :
a(param_n),
b(param_n),
c(param_n),
d(param_n),
e(param_n),
rmse(param_n),
decay(0.9),
mean_decay(0.9),
eps(.00001),
base_lr(start_lr_ - end_lr_),
end_lr(end_lr_),
lr_decay(pow(.5, 1.0 / half_time))
{

}

void Trainer::update(CudaVec *param, CudaVec &grad) {
	float lr = end_lr + base_lr;
	base_lr *= lr_decay;

	a = grad;
	a *= a;
	rmse *= decay;
	a *= (1.0 - decay);
	rmse += a;

	b = rmse;
	b.sqrt();
	b += eps;

	c = grad;
	c /= b;

	//SGD
	// net.grad *= .00001;
	// net.param += net.grad;

	//Marijn Trick

	//d = c;
	//d *= (1.0 - mean_decay);
	//e *= mean_decay;
	//e += d;

	//d = e;
	//d.abs();
	//c *= d;

	//Marijn Trick 2

	// if (epoch >= burnin) {
	//   d = param;
	//   d *= (1.0 / n_sums);
	//   e += d;
	//   ++sum_counter;

	//   if (sum_counter == n_sums) {
	//     param = e;
	//     e.zero();
	//     c.zero();
	//     sum_counter = 0;
	//     save("mean.net");
	//   }

	// }

	//Momentum

	//d = c;
	c *= (1.0 - mean_decay);
	e *= mean_decay;
	e += c;
	// c = e;

	//update
	//c.clip(1.);
	e *= lr;
	*param += e;

	print_last(grad.to_vector(), 10);
	print_last(rmse.to_vector(), 10);
	print_last(e.to_vector(), 10);
	print_last((*param).to_vector(), 10);
}