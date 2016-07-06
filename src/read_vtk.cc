#include "read_vtk.h"

#include <string>
#include <iostream>


#include <vtkMetaImageReader.h>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>
using namespace std;

Volume read_vtk(string filename) {
  Volume vol;

  vtkSmartPointer<vtkMetaImageReader> reader =
    vtkSmartPointer<vtkMetaImageReader>::New();
  reader->SetFileName(filename.c_str());
  reader->ReleaseDataFlagOn(); 
  reader->Update();
  cout << "width: " << reader->GetWidth() << endl;
  cout << "height: " << reader->GetHeight() << endl;
  
  //    cout << "height: " << reader->GetDepth() << endl;

  cout << "comp: " << reader->GetNumberOfComponents() << endl;
  cout << "pix repr: " << reader->GetPixelRepresentation() << endl;
  vtkImageData *data = reader->GetOutput();
  vector<int> dims(3);
  copy(data->GetDimensions(), data->GetDimensions() + dims.size(), &dims[0]);
  cout << "scalar type: " << data->GetScalarType() << endl;
  cout << "dims: " << dims << endl;
  return vol;
  for (size_t z(0); z < dims[2]; ++z)
    for (size_t y(0); y < dims[1]; ++y) {
      for (size_t x(0); x < dims[0]; ++x)
	cout << (int)*reinterpret_cast<unsigned char*>(data->GetScalarPointer(x, y, z)) << " ";
      cout << endl;
    }
				   
  return vol;
}
