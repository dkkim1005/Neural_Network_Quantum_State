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
  options.push_back(pair_t("Ni", "# of input nodes"));
  options.push_back(pair_t("Nh", "# of hidden nodes"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of trials to compute spontaneous magnetization"));
  options.push_back(pair_t("h", "transverse-field strength"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("lattice", "lattice type (CH,SQ,CB)"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "300"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("path", "."));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);
  const int nInputs = parser.find<int>("Ni"),
    nHiddens = parser.find<int>("Nh"),
    nChains = parser.find<int>("ns"),
    niter = parser.find<int>("niter"),
    nWarmup = parser.find<int>("nwarm"),
    nMonteCarloSteps = parser.find<int>("nms"),
    deviceNumber = parser.find<int>("dev"),
    ver = parser.find<int>("ver");
  const double h = parser.find<double>("h");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string path = parser.find<>("path");
  std::string hfstr = std::to_string(h);
  hfstr.erase(hfstr.find_last_not_of('0') + 1, std::string::npos);
  hfstr.erase(hfstr.find_last_not_of('.') + 1, std::string::npos);

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

  FFNN<double> psi(nInputs, nHiddens, nChains);

  const std::string filename = parser.find<>("lattice") + "-Ni" + parser.find<>("Ni") + "Nh"
    + parser.find<>("Nh") + "Hf" + hfstr + "V" + parser.find<>("ver");
  const std::string filepath = parser.find<>("path") + "/" + filename;

  // load parameters: w,a,b
  psi.load(FFNNDataType::W1, filepath + "Dw1.dat");
  psi.load(FFNNDataType::W2, filepath + "Dw2.dat");
  psi.load(FFNNDataType::B1, filepath + "Db1.dat");

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(niter)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(nInputs)*
    static_cast<unsigned long>(nChains);

  // measurements of the overlap integral for the given wave functions
  struct TRAITS { using AnsatzType = FFNN<double>; using FloatType = double; };

  Sampler4SpinHalf<TRAITS> smp(psi, seed, nBlocks);
  MeasSpontaneousMagnetization<TRAITS> smag(smp);
  double m1, m2, m4;
  smag.measure(niter, nMonteCarloSteps, nWarmup, m1, m2, m4);

  const std::string wfilename = parser.find<>("path") + "/smag-" + parser.find<>("lattice") + "-Ni" + parser.find<>("Ni") + ".dat";
  std::ofstream wfile;
  if(!std::ifstream(wfilename).is_open())
  {
    wfile.open(wfilename);
    wfile << "#   h      m1       m2       m4" << std::endl;
  }
  else
    wfile.open(wfilename, std::ofstream::app);
  wfile << std::setw(7) << h << " " << std::setw(7) << m1 << " " << std::setw(7) << m2 << " " << std::setw(7) << m4 << std::endl;
  wfile.close();

  return 0;
}
