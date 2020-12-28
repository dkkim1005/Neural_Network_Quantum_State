// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include "../include/common.cuh"
#include "../include/neural_quantum_state.cuh"
#include "../include/hamiltonians.cuh"
#include "../include/optimizer.cuh"
#include "../../cpu/include/argparse.hpp"

template <typename FloatType>
std::string remove_zeros_in_str(const FloatType val);

using namespace spinhalf;

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("L", "# of lattice sites"));
  options.push_back(pair_t("nf", "# of filters"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of iterations to train RBMTrSymm"));
  options.push_back(pair_t("alpha", "exponent in the two-body interaction: J_{i,j} ~ 1/|i-j|^{alpha}"));
  options.push_back(pair_t("theta", "J = sin(theta), h = -cos(theta)"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("lr", "learning_rate"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("ifprefix", "prefix of the file to load data"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("lr", "1e-2"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("ifprefix", "None"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int L = parser.find<int>("L"),
    nChains = parser.find<int>("ns"),
    nWarmup = parser.find<int>("nwarm"),
    nMonteCarloSteps = parser.find<int>("nms"),
    deviceNumber = parser.find<int>("dev"),
    nIterations =  parser.find<int>("niter");
  const double lr = parser.find<double>("lr");
  const unsigned long long seed = parser.find<unsigned long long>("seed");
  const std::string path = parser.find<>("path") + "/",
    Lstr = parser.find<>("L"),
    ifprefix = parser.find<>("ifprefix");
  const auto nFiltersArr = parser.mfind<int>("nf");
  const auto alphaArr = parser.mfind<double>("alpha");
  const auto verArr = parser.mfind<int>("ver");
  const auto thetaArr = parser.mfind<double>("theta");

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

  struct SamplerTraits { using AnsatzType = RBMTrSymm<double>; using FloatType = double; };

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(nIterations)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(L)*
    static_cast<unsigned long>(nChains);

  for (const auto & ver : verArr)
    for (const auto & nf : nFiltersArr)
      for (const auto & alpha : alphaArr)
        for (const auto & theta : thetaArr)
        {
          const std::string verstr = std::to_string(ver),
            nfstr = std::to_string(nf),
            alphastr = remove_zeros_in_str(alpha),
            thetastr = remove_zeros_in_str(theta);
          RBMTrSymm<double> machine(L, nf, nChains);
          const double J = std::sin(theta), h = -std::cos(theta);
          // load parameters
          const std::string prefix = path + "RBMTrSymmLICH-L" + Lstr + "NF" + nfstr + "A" + alphastr + "T" + thetastr + "V" + verstr;
          const std::string prefix0 = (ifprefix.compare("None")) ? path+ifprefix : prefix;
          machine.load(prefix0);
          // Transverse Field Ising Hamiltonian with long-range interaction on the 1D chain lattice
          LITFIChain<SamplerTraits> sampler(machine, L, h, J, alpha, true, seed, nBlocks, prefix);
          const auto start = std::chrono::system_clock::now();
          sampler.warm_up(nWarmup);
          StochasticReconfigurationCG<double> iTimePropagator(nChains, machine.get_nVariables());
          iTimePropagator.propagate(sampler, nIterations, nMonteCarloSteps, lr);
          // save parameters
          machine.save(prefix);
          const auto end = std::chrono::system_clock::now();
          std::chrono::duration<double> elapsed_seconds = end-start;
          std::cout << "# elapsed time: " << elapsed_seconds.count() << "(sec)" << std::endl;
        }

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
