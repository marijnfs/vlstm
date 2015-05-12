#ifndef __RAW_H__
#define __RAW_H__

#include "volume.h"
#include "util.h"
#include <string>
#include <iostream>

inline Volume open_raw(std::string name1, std::string name2, std::string name3, int W, int H, int Z)
{
    std::ifstream file1(name1.c_str(), std::ios::binary);
    std::ifstream file2(name2.c_str(), std::ios::binary);
    std::ifstream file3(name3.c_str(), std::ios::binary);

    if (!file1)
        throw StringException(name1.c_str());
    Volume volume(VolumeShape{Z, 3, W, H});
    std::vector<float> data(Z*3*W*H);
    std::vector<float>::iterator it(data.begin());



    for (int z(0); z < Z; z++){
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file1);
            
        }
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file2);
        }
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file3);
                      
        }
        
    }

    std::vector<float>::iterator it1(data.begin());
    for (int n(0); n < Z*3; n++, it1 += (W * H)){
        normalize(it1, it1 + (W * H));
    }
    volume.from_vector(data);

    return volume;

}

inline Volume open_raw(std::string name, int W, int H, int Z)
{
    int C = 5;
    std::ifstream file(name.c_str(), std::ios::binary);

    if (!file)
        throw StringException(name.c_str());

    Volume volume(VolumeShape{Z, C, W, H});
    std::vector<float> data(Z * C * W * H);

    for (int z(0); z < Z; z++){
        for (int i(0); i < W*H; i++){
            float value = byte_read<float>(file);
            int label(0);
            if (value == 5 || value == 6) label = 1;
            if (value == 1 || value == 2) label = 2;
            if (value == 3 || value == 4) label = 3;
            if (value == 7 || value == 8) label = 4;
            
            data[z*C*W*H + label * W*H + i] = 1;
            
        }
    }

    volume.from_vector(data);

    return volume;

}


#endif
