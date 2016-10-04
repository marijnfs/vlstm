#include <iostream>
//#include <cuda.h>
#include <sstream>
#include <algorithm>

#include "volume.h"
#include "volumenetwork.h"
#include "vlstm.h"
#include "handler.h"
//#include "tiff.h"
#include "log.h"
#include "divide.h"
#include "raw.h"
#include "trainer.h"
#include "walk.h"
#include "read_vtk.h"

using namespace std;



int main(int argc, char **argv) {
	//VolumeShape shape{100, 1, 512, 512};
	Handler::set_device(0);
	srand(time(0));
	
	string brain_path = "/home/marijnfs/data/BRATS2015_Training/processed";

	string ds = date_string();
	ostringstream oss;
	oss << "log/result-" << ds << ".log";
	Log logger(oss.str());

	ostringstream oss1;
	oss1 << "log/network-" << ds << ".net";
	string netname = oss1.str();

	vector<string> input_paths = walk(brain_path, ".vol", "joined");
 	vector<string> target_paths = walk(brain_path, ".vol", "seg");
	
	assert(input_paths.size() == target_paths.size());
	//Net parameters
	int kg(7), ko(7), c(1);

	Volume first_input(input_paths[0]);//////

	//TMP
	//Volume first_input(VolumeShape{4, 2, 4, 4});
	//first_input.init_normal(0.0, 1.0);
	
	// Volume tiff_data = open_tiff("7nm/input.tif", true);
	// // Volume outputs[brainnum] = open_tiff("7nm/input.tif", true);
	// Volume tiff_label = open_tiff("7nm/binary-labels.tif", false, true);


	

	VolumeShape sub_input_shape{64, first_input.shape.c, 64, 64};
	VolumeShape sub_target_shape{64, 5, 64, 64};


	VolumeNetwork net(sub_input_shape);

	//net.add_fc(8);
	//net.add_tanh();
	
	//net.add_vlstm(7, 7, 4);
	//net.add_vlstm(7, 7, 4);

	//net.add_fc(16);
	//net.add_tanh();
	//net.add_fc(16);
	//net.add_tanh();

	
	//net.add_hwv(7);
	//net.add_hwv(7);
	//net.add_hwv(7, 8);
	//net.add_hwv(7, 8);
	//net.add_hwv(7, 8);
	//	net.add_hwv(7, 8);
	//net.add_vlstm(7, 7, 4);
	//net.add_vlstm(7, 7, 4);
	//net.add_fc(2);
	//net.add_tanh();
	//net.add_fc(5);
	//net.add_fc(16);
	//net.add_tanh();
	net.add_vlstm(7, 7, 32);
	net.add_vlstm(7, 7, 32);
	//net.add_fc(16);
	//net.add_tanh();
	//net.add_fc(128);
	//	net.add_tanh();
	//net.add_fc(256);
	//net.add_tanh();
	//net.add_vlstm(5, 5, 4);
	//net.add_vlstm(5, 5, 4);
	//net.add_fc(8);
	//net.add_tanh();
	//net.add_vlstm(5, 5, 4);
	//net.add_vlstm(5, 5, 4);
	//net.add_vlstm(3, 3, 8);
	//net.add_vlstm(3, 3, 5);
	net.add_fc(5);
	net.add_softmax();
	net.finish();

	if (argc > 1)
	  net.load(argv[1]);
	else
	  //net.init_normal(0.0, .1);
	  net.param_vec.init_normal(0.0, .05);
	logger << "begin description\n";
	//logger << "subvolume shape " << sub_shape << "\n";
	net.describe(logger.file);
	logger << "end description\n";

	Trainer trainer(net.param_vec.n, .01, 0.00001, 400, .99);

	int iteration(0);
	Volume input, target;
	Volume sub_target(sub_target_shape);

	while (true) {
	  vector<int> indices(input_paths.size());
	  for (size_t i(0); i < indices.size(); ++i) indices[i] = i;
	  
	  random_shuffle(indices.begin(), indices.end());
	  
	  for (int index : indices) {
	    Timer timer;
	    cout << "iteration " << iteration << ", vol: " << input_paths[index] << " " << target_paths[index] << endl;
	    
	    input.load_file(input_paths[index]);
	    target.load_file(target_paths[index]);

	    copy_subvolume(input, net.input(), target, sub_target, false, rand()%2, false, false); // rotate, xflip, yflip, zflip
	    
	    
	    /*	    ////TEMP
	    Volume target(first_input.shape);
	    target.init_normal(0.0, 1.0);
	    net.input().from_volume(first_input);
	    net.forward();
	    net.calculate_loss(target);
	    net.backward();


	    //cout << first_input.to_vector() << endl;
	    //cout << target.to_vector() << endl;
	    //return 0;
	    auto v = net.param_vec.to_vector();
	    auto g = net.grad_vec.to_vector();
	    
	    float eps(.002);
	    for (int i(0); i < v.size(); ++i) {
	      auto v_copy = v;
	      v_copy[i] = v[i] + eps;
	      net.param_vec.from_vector(v_copy);
	      net.forward();
	      float loss1 = net.calculate_loss(target);

	      v_copy[i] = v[i] - eps;
	      net.param_vec.from_vector(v_copy);
	      net.forward();
	      float loss2 = net.calculate_loss(target);
	      float est = (loss2-loss1) / (2.0 * eps);
	      //if (abs(est - g[i])/(abs(g[i]) + eps) > 1.)
		cout << i << ": " << g[i] << " " << est << endl;//" " << loss1 << " " << loss2 << endl;
	    }
	      
	    return 0;
	    ////TEMP END
	    */
	    
	    //cout << target.to_vector() << endl;
	    //target.draw_volume("t0_%i.png", 0);
	    //target.draw_volume("t1_%i.png", 1);
	    //target.draw_volume("t2_%i.png", 2);
	    //target.draw_volume("t3_%i.png", 3);
	    //target.draw_volume("t4_%i.png", 4);
	    //return 0;

	    //cout << "param: " << net.param_vec.to_vector() << endl;
	    //net.input().draw_volume("in_%i.png", 0);
	    //net.input().from_volume(input);
	    net.forward();

	    //Draw Stuff
	    //net.output().draw_volume("output0_%i.png", 0);
	    //target.draw_volume("target0_%i.png", 0);
	    //net.output().draw_volume("output1_%i.png", 1);
	    //target.draw_volume("target1_%i.png", 1);
	    net.output().draw_volume("output4_%i.png", 4);
	    sub_target.draw_volume("target4_%i.png", 4);
	    return 0;
	    cout << "input: ";
	    print_wide(net.input().to_vector(), 101);
	    cout << "target: ";
	    print_wide(sub_target.to_vector(), 101);
	    cout << "output: ";
	    print_wide(net.output().to_vector(), 101);
	    //cout << "vol2: ";
	    // {
	    //ostringstream oss;
	      //  oss << "t_" << index << "-%i.png";
	      //target.draw_volume(oss.str(), 0);
	    //}
	  //print_wide(net.volumes[net.volumes.size()-2]->x.to_vector(), 41);
	    //float loss = net.calculate_loss(target);
	    float loss = net.calculate_loss(sub_target);
	    cout << "loss: " << loss << endl;

	    net.backward();
	    //cout << "grad: ";
	    //print_wide(net.grad_vec.to_vector(), 20); //Slow
	    //cout << "param: ";
	    //print_wide(net.param_vec.to_vector(), 20); //Slow
	    
	    trainer.update(&net.param_vec, net.grad_vec);

	    ++iteration;
	    cout << "Took: " << timer.since() << " seconds" << endl;
	    net.save(netname);
	  }
	}

}
