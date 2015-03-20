#ifndef __VOLUMENETWORK_H__
#define __VOLUMENETWORK_H__

struct VolumeNetwork {
	VolumeNetwork(VolumeShape shape);
	void add_vlstm(int kg, int ko, int c);
	void add_fc(int c);
	void add_softmax();

	std::vector<Parametrised<F>*> params;

	std::vector<VolumeOperation*> operations;
	std::vector<VolumeSet*> volumes;
	std::vector<VolumeShape> shapes;
};

#endif