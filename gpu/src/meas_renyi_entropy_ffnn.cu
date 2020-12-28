// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include "../include/meas.cuh"
#include "../../cpu/include/argparse.hpp"

using namespace spinhalf;

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("L", "# of lattice sites"));
  options.push_back(pair_t("nh", "# of hidden nodes"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of iterations to measure Renyi entropy"));
  options.push_back(pair_t("l", "length of the subregion"));
  options.push_back(pair_t("h", "transverse-field strength"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("path", "directory to load files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("ifprefix", "prefix of the file to load data"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("ifprefix", "None"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int L = parser.find<int>("L"),
    nInputs = L,
    nHiddens = parser.find<int>("nh"),
    nChains = parser.find<int>("ns"),
    nWarmup = parser.find<int>("nwarm"),
    nMonteCarloSteps = parser.find<int>("nms"),
    deviceNumber = parser.find<int>("dev"),
    nIterations =  parser.find<int>("niter"),
    subRegionLength = parser.find<int>("l"),
    version = parser.find<int>("ver");
  const double h = parser.find<double>("h");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string path = parser.find<>("path") + "/",
    nistr = std::to_string(nInputs),
    nhstr = std::to_string(nHiddens),
    vestr = std::to_string(version),
    ifprefix = parser.find<>("ifprefix");
  std::string hfstr = std::to_string(h);
  hfstr.erase(hfstr.find_last_not_of('0') + 1, std::string::npos);
  hfstr.erase(hfstr.find_last_not_of('.') + 1, std::string::npos);

  if (subRegionLength < 0 || subRegionLength >= L)
  {
    std::cout << subRegionLength << std::endl;
    std::cerr << "# error : The subregion length should be within the range: 0 <= l < L-1" << std::endl;
    exit(1);
  }

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

  FFNN<double> psi1(nInputs, nHiddens, nChains), psi2(nInputs, nHiddens, nChains), psi3(nInputs, nHiddens, nChains);

  // load parameters
  const std::string prefix = path + "CH-Ni" + nistr + "Nh" + nhstr + "Hf" + hfstr + "V" + vestr;
  const std::string prefix0 = (ifprefix.compare("None")) ? path+ifprefix : prefix;

  psi1.load(FFNNDataType::W1, prefix0 + "Dw1.dat");
  psi1.load(FFNNDataType::W2, prefix0 + "Dw2.dat");
  psi1.load(FFNNDataType::B1, prefix0 + "Db1.dat");
  psi1.copy_to(psi2);
  psi1.copy_to(psi3);

  struct TRAITS { using AnsatzType = FFNN<double>; using FloatType = double; };

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(nIterations)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(nInputs)*
    static_cast<unsigned long>(nChains);

  Sampler4SpinHalf<TRAITS> sampler1(psi1, seed, nBlocks), sampler2(psi2, seed+987654321ul, nBlocks);
  MeasRenyiEntropy<TRAITS> S2measure(sampler1, sampler2, psi3);
  S2measure.measure(subRegionLength, nIterations, nMonteCarloSteps, nWarmup);

  return 0;
}
