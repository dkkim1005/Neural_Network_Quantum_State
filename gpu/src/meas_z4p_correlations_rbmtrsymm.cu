#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <fstream>
#include "../include/meas.cuh"
#include "../../cpu/include/argparse.hpp"

using namespace spinhalf;

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("L", "# of lattice sites"));
  options.push_back(pair_t("nf", "# of filters"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of trials to compute 4-points spin correlation"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("filename", "data file of the trained model"));
  options.push_back(pair_t("tag", "tag of results"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("tag", "0"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);
  const int L = parser.find<int>("L"),
    nf = parser.find<int>("nf"),
    nChains = parser.find<int>("ns"),
    niter = parser.find<int>("niter"),
    nWarmup = parser.find<int>("nwarm"),
    nMonteCarloSteps = parser.find<int>("nms"),
    deviceNumber = parser.find<int>("dev");
  const unsigned long seed = parser.find<unsigned long>("seed");

  // print info of the registered args
  parser.print(std::cout);

  // check whether the cuda device is available
  int devicesCount;
  CHECK_ERROR(cudaSuccess, cudaGetDeviceCount(&devicesCount));
  if (deviceNumber >= devicesCount)
  {
    std::cerr << "# error ---> dev(" << deviceNumber << ") >= # of devices(" << devicesCount << ")" << std::endl;
    exit(1);
  }
  CHECK_ERROR(cudaSuccess, cudaSetDevice(deviceNumber));

  RBMTrSymm<float> psi(L, nf, nChains);

  const std::string filepath = parser.find<>("path") + "/" + parser.find<>("filename");

  // load parameters: w,a,b
  psi.load(filepath);

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(niter)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(L)*
    static_cast<unsigned long>(nChains);

  // measurements of the overlap integral for the given wave functions
  struct TRAITS { using AnsatzType = RBMTrSymm<float>; using FloatType = float; };

  Sampler4SpinHalf<TRAITS> smp(psi, seed, nBlocks);
  Meas4PointsSpinZCorrelation<TRAITS> corr(smp);
  std::vector<float> sz4(L*L, 0);
  corr.measure(niter, nMonteCarloSteps, nWarmup, sz4.data());

  std::ofstream wfile(parser.find<>("path") + "/Corr-Z4-" + parser.find<>("filename") + "-TAG" + parser.find<>("tag")+ ".dat");
  for (int i=0; i<L; ++i)
  {
    for (int j=0; j<L; ++j)
      wfile << sz4[i*L+j] << " ";
    wfile << std::endl;
  }
  wfile.flush();
  wfile.close();

  return 0;
}
