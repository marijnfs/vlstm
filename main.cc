#include "volume.h"
#include "vlstm.h"


int main() {
	//VolumeShape shape{100, 1, 512, 512};
	VolumeShape shape{100, 1, 100, 100};
	int kg(3), ko(3), c(1);
	VLSTM vlstm(shape, kg, ko, c);
	vlstm.forward();
	vlstm.backward();
}
