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
       << NBFMM_VERSION_MAJOR << "."
       << NBFMM_VERSION_MINOR << "."
       << NBFMM_VERSION_PATCH << " demo" << endl;
  Solver solver(4, 1024, make_float4(0, 0, 1, 1));
  return 0;
}
