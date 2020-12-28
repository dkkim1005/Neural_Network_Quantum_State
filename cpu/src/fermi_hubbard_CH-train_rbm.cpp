// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include <chrono>
#include "../include/hamiltonians.hpp"
#include "../include/optimizer.hpp"
#include "../include/argparse.hpp"

using namespace spinhalf;

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("N", "# of lattice sites (2 x nInputs)"));
  options.push_back(pair_t("al", "ratio of hidden nodes to input nodes"));
  options.push_back(pair_t("niter", "# of iterations"));
  options.push_back(pair_t("nms", "# of montecarlo steps"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("np", "# of particles"));
  options.push_back(pair_t("nthread", "# of threads for openmp"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("lr", "learning rate"));
  options.push_back(pair_t("t", "hopping element"));
  options.push_back(pair_t("U", "onsite interaction"));
  options.push_back(pair_t("pbc", "use periodic boundary condition (true : 1 or false : 0)"));
  options.push_back(pair_t("prefix", "filename to load and save"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  // env; default value
  defaults.push_back(pair_t("nthread", "1"));
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("lr", "1e-3"));
  defaults.push_back(pair_t("t", "1"));
  defaults.push_back(pair_t("prefix", "./"));
  defaults.push_back(pair_t("seed", "0"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int nInputs = 2*parser.find<int>("N"),
    nHiddens = static_cast<int>(nInputs*parser.find<double>("al")),
    nChains = parser.find<int>("ns"),
    nParticles = parser.find<int>("np"),
    niter = parser.find<int>("niter"),
    nms = parser.find<int>("nms"),
    nwarm = parser.find<int>("nwarm"),
    num_omp_threads = parser.find<int>("nthread");
  const double lr = parser.find<double>("lr"),
    t = parser.find<double>("t"),
    U = parser.find<double>("U");
  const bool usePBC = parser.find<bool>("pbc");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string prefix = parser.find<>("prefix");

  // print info of the registered args
  parser.print(std::cout);

  // set number of threads for openmp
  omp_set_num_threads(num_omp_threads);

  RBM<double> machine(nInputs, nHiddens, nChains);

  // load parameters: w,a,b
  machine.load(RBMDataType::W, prefix + "w.dat");
  machine.load(RBMDataType::V, prefix + "v.dat");
  machine.load(RBMDataType::H, prefix + "h.dat");

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(niter)*
    static_cast<unsigned long>(nms)*static_cast<unsigned long>(nInputs)*
    static_cast<unsigned long>(nChains);

  struct TRAITS
  {
    typedef RBM<double> AnsatzType; 
    typedef double FloatType;
  };

  fermion::jordanwigner::HubbardChain<TRAITS> sampler(machine, U, t, nParticles, usePBC, nBlocks);

  sampler.warm_up(nwarm);

  StochasticReconfigurationCG<double> optimizer(nChains, machine.get_nVariables());
  optimizer.propagate(sampler, niter, nms, lr);

  // save parameters: w,a,b
  machine.save(RBMDataType::W, prefix + "w.dat");
  machine.save(RBMDataType::V, prefix + "v.dat");
  machine.save(RBMDataType::H, prefix + "h.dat");

  return 0;
}
