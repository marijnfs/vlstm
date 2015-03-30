#ifndef __TIFF_H__
#define __TIFF_H__

#include "volume.h"
#include <tiffio.h>
#include <string>
#include <iostream>

inline Volume open_tiff(std::string name, bool do_normalize = false, bool binary = false)
{
    TIFF* tif = TIFFOpen(name.c_str(), "r");
    int z = 0;
   	uint32 w(0), h(0);

   	std::vector<std::vector<float> > v_data;
    do {
        size_t npixels;

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        npixels = w * h;

		uint32 *raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));

        std::vector<float> fdata;
        fdata.reserve(npixels);
        z++;
        if (TIFFReadRGBAImage(tif, w, h, raster, 0)) {
			//for (auto d : data)
			//	std::cout << int(d) << " ";
        	for (size_t i(0); i < npixels; ++i) {
				uint32 val(raster[i]);
				val = val % (1 << 8);
                float fval = static_cast<float>(val) / 255.;
                fdata.push_back(fval);
			}
			_TIFFfree(raster);
        } else {
        	throw "fail";
        }
        v_data.push_back(fdata);

	if (z == 5) //hack
         	break;

    } while (TIFFReadDirectory(tif));

	if (do_normalize) {
		for (auto &v : v_data)
			normalize(&v);
	}

    if (binary) {
        Volume v(VolumeShape{z, 2, w, h});
        for (size_t i(0); i < v_data.size(); ++i) {
            handle_error( cudaMemcpy(v.slice(i) + v_data[i].size(), &v_data[i][0], v_data[i].size() * sizeof(F), cudaMemcpyHostToDevice));
            for (auto &v : v_data[i])
                v = 1.0 - v;
            handle_error( cudaMemcpy(v.slice(i), &v_data[i][0], v_data[i].size() * sizeof(F), cudaMemcpyHostToDevice));
        }
        TIFFClose(tif);
        return v;
    } else {
        Volume v(VolumeShape{z, 1, w, h});
        for (size_t i(0); i < v_data.size(); ++i)
    	   handle_error( cudaMemcpy(v.slice(i), &v_data[i][0], v_data[i].size() * sizeof(F), cudaMemcpyHostToDevice));
        std::cout << z << std::endl;
        TIFFClose(tif);
        return v;
    }
}

#endif
