#ifndef __TIFF_H__
#define __TIFF_H__

#include "volume.h"
#include <tiffio.h>
#include <string>
#include <iostream>

inline Volume open_tiff(std::string name)
{
    TIFF* tif = TIFFOpen(name.c_str(), "r");
    int dircount = 0;
   	uint32 w(0), h(0);

   	std::vector<std::vector<float> > v_data;
    do {
        size_t npixels;
        uint32* raster;

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        npixels = w * h;

        std::vector<uint8> data(npixels * 4);
        std::vector<float> fdata;
        fdata.reserve(npixels);
        dircount++;
        if (TIFFReadRGBAImage(tif, w, h, (uint32*) &data[0], 0)) {
        	std::cout << int(data[0]) << " " << int(data[1]) << " " << int(data[2]) << " " << int(data[3]) << std::endl;
        	for (size_t i(0); i < data.size(); i += 4)
	        	fdata.push_back(static_cast<float>(data[i]) / 255.);
        } else {
        	throw "fail";
        }
        v_data.push_back(fdata);
    } while (TIFFReadDirectory(tif));

    Volume v(VolumeShape{dircount, 1, w, h});
    for (size_t i(0); i < v_data.size(); ++i)
    	handle_error( cudaMemcpy(v.slice(i), &v_data[0], v_data[0].size() * sizeof(F), cudaMemcpyHostToDevice));
    std::cout << dircount << std::endl;
    TIFFClose(tif);
    return v;
}

#endif
