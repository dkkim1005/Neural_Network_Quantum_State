// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include "../include/common.cuh"
#include "../include/neural_quantum_state.cuh"
#include "../include/hamiltonians.cuh"
#include "../include/optimizer.cuh"
#include "../../cpu/include/argparse.hpp"

using namespace spinhalf;

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("L", "# of lattice sites"));
  options.push_back(pair_t("nh", "# of hidden nodes"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of iterations to train RBM"));
  options.push_back(pair_t("h", "transverse-field strength"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("J", "coupling constant"));
  options.push_back(pair_t("lr", "learning_rate"));
  options.push_back(pair_t("rsd", "cutoff value of the energy deviation per energy (convergence criterion)"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("ifprefix", "prefix of the file to load data"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("J", "-1.0"));
  defaults.push_back(pair_t("lr", "5e-3"));
  defaults.push_back(pair_t("rsd", "1e-3"));
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
    version = parser.find<int>("ver");
  const double J = parser.find<double>("J"),
    lr = parser.find<double>("lr"),
    RSDcutoff = parser.find<double>("rsd");
  const unsigned long long seed = parser.find<unsigned long long>("seed");
  const std::string path = parser.find<>("path") + "/",
    nistr = std::to_string(nInputs),
    nhstr = std::to_string(nHiddens),
    vestr = std::to_string(version),
    ifprefix = parser.find<>("ifprefix");
  const auto hfields = parser.mfind<double>("h");

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

  struct SamplerTraits { using AnsatzType = RBM<double>; using FloatType = double; };

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(nIterations)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(nInputs)*
    static_cast<unsigned long>(nChains);

  for (int i=0; i<hfields.size(); ++i)
  {
    const double h = hfields[i];
    if (hfields.size() > 1)
      std::cout << "# h: " << h << " --- job id:" << (i+1) << " / " << hfields.size() << std::endl;
    std::string hfstr = std::to_string(h);
    hfstr.erase(hfstr.find_last_not_of('0') + 1, std::string::npos);
    hfstr.erase(hfstr.find_last_not_of('.') + 1, std::string::npos);

    RBM<double> machine(nInputs, nHiddens, nChains);

    // load parameters
    const std::string prefix = path + "CH-Nv" + nistr + "Nh" + nhstr + "Hf" + hfstr + "V" + vestr;
    const std::string prefix0 = (ifprefix.compare("None")) ? path+ifprefix : prefix;

    machine.load(prefix0);

    // Transverse Field Ising Hamiltonian on the 1D chain lattice
    TFIChain<SamplerTraits> sampler(machine, L, h, J, seed, nBlocks, prefix);

    const auto start = std::chrono::system_clock::now();

    sampler.warm_up(nWarmup);
    StochasticReconfigurationCG<double> iTimePropagator(nChains, machine.get_nVariables());
    iTimePropagator.propagate(sampler, nIterations, nMonteCarloSteps, lr, RSDcutoff);

    // save parameters
    machine.save(prefix);

    const auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "# elapsed time: " << elapsed_seconds.count() << "(sec)" << std::endl;
  }

  return 0;
}
