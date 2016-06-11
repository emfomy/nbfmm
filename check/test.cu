////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/test.cu
/// @brief   The test code
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResult.h>

using namespace CppUnit;

int main() {
  TextUi::TestRunner runner;
  runner.addTest(TestFactoryRegistry::getRegistry("Solver").makeTest());
  BriefTestProgressListener linster;
  runner.eventManager().addListener(&linster);
  runner.run("", false, true, false);
  return 0;
}
