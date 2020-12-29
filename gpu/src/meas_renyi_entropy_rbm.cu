// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include <iomanip>
#include "../include/meas.cuh"
#include "../../cpu/include/argparse.hpp"

template <typename FloatType>
std::string remove_zeros_in_str(const FloatType val);

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
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("path", "directory to load files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("file", "file name to load data"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("seed", "0"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int L = parser.find<int>("L"),
    nChains = parser.find<int>("ns"),
    nWarmup = parser.find<int>("nwarm"),
    nMonteCarloSteps = parser.find<int>("nms"),
    deviceNumber = parser.find<int>("dev"),
    nIterations =  parser.find<int>("niter");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string path = parser.find<>("path") + "/",
    Lstr = parser.find<>("L");
  const auto nHiddensArr = parser.mfind<int>("nh");
  const auto lArr = parser.mfind<int>("l");

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

  struct TRAITS { using AnsatzType = RBM<double>; using FloatType = double; };

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(nIterations)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(L)*
    static_cast<unsigned long>(nChains);

  const std::string filename = path + parser.find<>("file") + "-renyi_entropy.dat";
  std::ofstream wfile;
  if (!std::ifstream(filename).is_open())
  {
    wfile.open(filename);
    wfile << "#        l       S_2           L" << std::endl;
  }
  else
    wfile.open(filename, std::fstream::app);
  wfile << std::setprecision(5);

  for (const auto & nHiddens : nHiddensArr)
    for (const auto & l : lArr)
    {
      if (l < 0 || l >= L)
      {
        std::cout << l << std::endl;
        std::cerr << "# error : The subregion length should be within the range: 0 <= l < L-1" << std::endl;
        exit(1);
      }
      RBM<double> psi1(L, nHiddens, nChains), psi2(L, nHiddens, nChains), psi3(L, nHiddens, nChains);
      // load parameters
      psi1.load(path + parser.find<>("file"));
      psi1.copy_to(psi2);
      psi1.copy_to(psi3);
      Sampler4SpinHalf<TRAITS> sampler1(psi1, seed, nBlocks), sampler2(psi2, seed+987654321ul, nBlocks);
      MeasRenyiEntropy<TRAITS> S2measure(sampler1, sampler2, psi3);
      const double S_2 = S2measure.measure(l, nIterations, nMonteCarloSteps, nWarmup);
      wfile << std::setw(10) << l << " "
            << std::setw(10) << S_2 << " "
            << std::setw(10) << L << std::endl;
    }

  wfile.close();

  return 0;
}

template <typename FloatType>
std::string remove_zeros_in_str(const FloatType val)
{
  std::string tmp = std::to_string(val);
  tmp.erase(tmp.find_last_not_of('0') + 1, std::string::npos);
  tmp.erase(tmp.find_last_not_of('.') + 1, std::string::npos);
  return tmp;
}
