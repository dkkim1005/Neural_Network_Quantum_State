// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include "../include/common.cuh"
#include "../include/neural_quantum_state.cuh"
#include "../include/hamiltonians.cuh"
#include "../include/optimizer.cuh"
#include "../../cpu/include/argparse.hpp"

template <typename FloatType>
std::string remove_zeros_in_str(const FloatType val);

using namespace spinhalf;
using namespace fermion::jordanwigner;

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("L", "# of lattice sites (2 x nInputs)"));
  options.push_back(pair_t("al", "ratio of hidden nodes to input nodes"));
  options.push_back(pair_t("niter", "# of iterations"));
  options.push_back(pair_t("nms", "# of montecarlo steps"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("np", "# of particles: up, down"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("lr", "learning rate"));
  options.push_back(pair_t("t", "hopping element"));
  options.push_back(pair_t("U", "onsite interaction"));
  options.push_back(pair_t("pbc", "use periodic boundary condition (true : 1 or false : 0)"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  // env; default value
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("lr", "1e-2"));
  defaults.push_back(pair_t("t", "1"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("seed", "0"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int nInputs = 2*parser.find<int>("L"),
    nHiddens = static_cast<int>(nInputs*parser.find<double>("al")),
    nChains = parser.find<int>("ns"),
    niter = parser.find<int>("niter"),
    nms = parser.find<int>("nms"),
    nwarm = parser.find<int>("nwarm");
  const auto np = parser.mfind<int, 2>("np");
  const double lr = parser.find<double>("lr"),
    t = parser.find<double>("t"),
    U = parser.find<double>("U");
  const bool usePBC = parser.find<bool>("pbc");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string path = parser.find<>("path") + "/";

  // print info of the registered args
  parser.print(std::cout);

  RBM<double> machine(nInputs, nHiddens, nChains);

  // load parameters
  const std::string prefix = path
    + "RBM-Hubbard-L" + parser.find<>("L")
    + "AL" + parser.find<>("al")
    + "NP" + parser.find<>("np")
    + "U" + remove_zeros_in_str(U)
    + "V" + parser.find<>("ver");

  // load parameters: w,a,b
  machine.load(prefix);

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(niter)*
    static_cast<unsigned long>(nms)*static_cast<unsigned long>(nInputs)*
    static_cast<unsigned long>(nChains);

  struct SamplerTraits { using AnsatzType = RBM<double>; using FloatType = double; };

  HubbardChain<SamplerTraits> sampler(machine, U, t, np, usePBC, seed, nBlocks, prefix);

  const auto start = std::chrono::system_clock::now();
  sampler.warm_up(nwarm);

  StochasticReconfigurationCG<double> iTimePropagator(nChains, machine.get_nVariables());
  iTimePropagator.propagate(sampler, niter, nms, lr);

  // save parameters: w,a,b
  machine.save(prefix);

  const auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "# elapsed time: " << elapsed_seconds.count() << "(sec)" << std::endl;

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
