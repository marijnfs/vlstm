#include "read_niftii.h"

#include <algorithm>
#include "util.h"

using namespace std;

NiftiVolume::NiftiVolume(string filename) {
  std::ifstream in_file(filename);
    
  hdr = byte_read<nifti_1_header>(in_file);
  
  fprintf(stderr, "\n%s header information:",filename.c_str());
  fprintf(stderr, "\nXYZT dimensions: %d %d %d %d",hdr.dim[0],hdr.dim[1],hdr.dim[2],hdr.dim[3],hdr.dim[4]);
  fprintf(stderr, "\nDatatype code and bits/pixel: %d %d",hdr.datatype,hdr.bitpix);
  fprintf(stderr, "\nScaling slope and intercept: %.6f %.6f",hdr.scl_slope,hdr.scl_inter);
  fprintf(stderr, "\nByte offset to data in datafile: %ld",(long)(hdr.vox_offset));
  fprintf(stderr, "\n");

  data.resize(hdr.dim[1] * hdr.dim[2] * hdr.dim[3] * hdr.dim[4]);

  assert(hdr.datatype == 4);
  vector<uint16_t> uint16_data(data.size());
  in_file.seekg(hdr.vox_offset, ios_base::beg);  
  in_file.read(reinterpret_cast<char*>(&uint16_data[0]), sizeof(uint16_t) * data.size());

  copy(uint16_data.begin(), uint16_data.end(), data.begin());

  vector<bool> mask(data.size());
  for (int n(0); n < mask.size(); ++n) if (uint16_data[n] != 0) mask[n] = true;  
  //normalize_masked(&data, mask);
  normalize(&data);
  // for (size_t i(0); i < data.size(); ++i)
  //cout << data[i] << " ";
  //  cout << endl;
}

Volume NiftiVolume::get_volume() {
  Volume volume(VolumeShape{hdr.dim[3], 1, hdr.dim[1], hdr.dim[2]});
  //Volume volume(VolumeShape{hdr.dim[3], 1, hdr.dim[2], hdr.dim[1]});
  //Volume volume(VolumeShape{hdr.dim[2], 1, hdr.dim[1], hdr.dim[3]});
  volume.from_vector(data);
  return volume;
}
