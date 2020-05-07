// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include "measurements.cuh"
#include "hamiltonians.cuh"
#include "../cpu/argparse.hpp"

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("L", "# of lattice sites"));
  options.push_back(pair_t("nh", "# of hidden nodes"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of iterations to sample the ground energy"));
  options.push_back(pair_t("h", "transverse-field strength"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("J", "coupling constant"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("ifprefix", "prefix of the file to load data"));
  options.push_back(pair_t("lattice", "lattice type(=CH,SQ,TRI,CB)"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("J", "-1.0"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("ifprefix", "None"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const uint32_t L = parser.find<uint32_t>("L"),
    nInputs = L*L,
    nHiddens = parser.find<uint32_t>("nh"),
    nChains = parser.find<uint32_t>("ns"),
    nWarmup = parser.find<uint32_t>("nwarm"),
    nMonteCarloSteps = parser.find<uint32_t>("nms"),
    deviceNumber = parser.find<uint32_t>("dev"),
    nIterations =  parser.find<uint32_t>("niter"),
    version = parser.find<uint32_t>("ver");
  const double h = parser.find<double>("h"), J = parser.find<double>("J");
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
  const std::string lattice = parser.find<>("lattice") + "-",
    prefix = path + lattice + "Ni" + nistr + "Nh" + nhstr + "Hf" + hfstr + "V" + vestr;
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
  using SamplerType = spinhalf::TFISQ<SamplerTraits>;
  SamplerType sampler(machine, L, h, J, seed, nBlocks);

  const auto start = std::chrono::system_clock::now();
  sampler.warm_up(nWarmup);

  const double groundEnergy = meas_energy<SamplerType, double>(sampler, nIterations, nWarmup, nMonteCarloSteps);

  std::cout << std::setprecision(7) << groundEnergy << std::endl;

  const auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "# elapsed time: " << elapsed_seconds.count() << "(sec)" << std::endl;

  return 0;
}
