////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/test.cu
/// @brief   The test code
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <iostream>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResult.h>
#include <nbfmm/core.hpp>

using namespace std;
using namespace CppUnit;

int main() {
  cout << "NBFMM "
       << NBFMM_VERSION_MAJOR << "."
       << NBFMM_VERSION_MINOR << "."
       << NBFMM_VERSION_PATCH << " check" << endl << endl;

  TextUi::TestRunner runner;
  runner.addTest(TestFactoryRegistry::getRegistry("Solver").makeTest());
  BriefTestProgressListener linster;
  runner.eventManager().addListener(&linster);
  runner.run("", false, true, false);
  return 0;
}
