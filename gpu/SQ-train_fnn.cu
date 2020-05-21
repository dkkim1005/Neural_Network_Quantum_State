// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include "common.cuh"
#include "neural_quantum_state.cuh"
#include "hamiltonians.cuh"
#include "optimizer.cuh"
#include "../cpu/argparse.hpp"

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("L", "# of lattice sites"));
  options.push_back(pair_t("nh", "# of hidden nodes"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of iterations to train FNN"));
  options.push_back(pair_t("h", "transverse-field strength"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("J", "coupling constant"));
  options.push_back(pair_t("lr", "learning_rate"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("ifprefix", "prefix of the file to load data"));
  options.push_back(pair_t("dr", "dropout rate"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("J", "-1.0"));
  defaults.push_back(pair_t("lr", "5e-3"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("ifprefix", "None"));
  defaults.push_back(pair_t("dr", "5e-1"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const uint32_t L = parser.find<uint32_t>("L"),
    nInputs = L*L,
    nHiddens = parser.find<int>("nh"),
    nChains = parser.find<int>("ns"),
    nWarmup = parser.find<int>("nwarm"),
    nMonteCarloSteps = parser.find<int>("nms"),
    deviceNumber = parser.find<int>("dev"),
    nIterations =  parser.find<int>("niter"),
    version = parser.find<int>("ver");
  const double h = parser.find<double>("h"),
    J = parser.find<double>("J"),
    lr = parser.find<double>("lr"),
    dr = parser.find<double>("dr");
  const uint64_t seed = parser.find<uint64_t>("seed");
  const std::string path = parser.find<>("path") + "/",
    nistr = std::to_string(nInputs),
    nhstr = std::to_string(nHiddens),
    vestr = std::to_string(version),
    ifprefix = parser.find<>("ifprefix");
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

  ComplexFNN<double> machine(nInputs, nHiddens, nChains);

  // load parameters
  const std::string prefix = path + "SQ-Ni" + nistr + "Nh" + nhstr + "Hf" + hfstr + "V" + vestr;
  const std::string prefix0 = (ifprefix.compare("None")) ? path+ifprefix : prefix;

  machine.load(FNNDataType::W1, prefix0 + "Dw1.dat");
  machine.load(FNNDataType::W2, prefix0 + "Dw2.dat");
  machine.load(FNNDataType::B1, prefix0 + "Db1.dat");

  struct SamplerTraits { using AnsatzType = ComplexFNN<double>; using FloatType = double;};

  // block size for the block splitting scheme of parallel Monte-Carlo
  const uint64_t nBlocks = static_cast<uint64_t>(nIterations)*
                           static_cast<uint64_t>(nMonteCarloSteps)*
                           static_cast<uint64_t>(nInputs)*
                           static_cast<uint64_t>(nChains);

  // Transverse Field Ising Hamiltonian on the square lattice
  spinhalf::TFISQ<SamplerTraits> sampler(machine, L, h, J, seed, nBlocks, dr, prefix);

  const auto start = std::chrono::system_clock::now();

  sampler.warm_up(nWarmup);

  const int nCutHiddens = static_cast<int>(nHiddens*dr);
  const int nVariables = nInputs*nCutHiddens + 2*nCutHiddens;
  StochasticReconfigurationCG<double> iTimePropagator(nChains, nVariables);
  iTimePropagator.propagate(sampler, nIterations, nMonteCarloSteps, lr);

  // save parameters
  machine.save(FNNDataType::W1, prefix + "Dw1.dat");
  machine.save(FNNDataType::W2, prefix + "Dw2.dat");
  machine.save(FNNDataType::B1, prefix + "Db1.dat");

  const auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "# elapsed time: " << elapsed_seconds.count() << "(sec)" << std::endl;

  return 0;
}
