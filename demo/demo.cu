////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    demo/demo.cu
/// @brief   The demo code
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <iostream>
#include <nbfmm.hpp>

using namespace std;
using namespace nbfmm;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main function
///
int main( int argc, char *argv[] ) {
  cout << "NBFMM "
       << NBFMM_VERSION_MAJOR_STRING << "."
       << NBFMM_VERSION_MINOR_STRING << "."
       << NBFMM_VERSION_PATCH_STRING << " demo" << endl;
  Solver solver(4, 1024);
  return 0;
}
