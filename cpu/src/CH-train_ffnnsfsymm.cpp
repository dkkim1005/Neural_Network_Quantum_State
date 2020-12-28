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
  options.push_back(pair_t("ninput", "# of input nodes"));
  options.push_back(pair_t("alpha", "# of filters"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of iterations to train FFNN"));
  options.push_back(pair_t("h", "transverse-field strength"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("J", "coupling constant"));
  options.push_back(pair_t("lr", "learning_rate"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("nthread", "# of threads for openmp"));
  options.push_back(pair_t("ifprefix", "prefix of the file to load data"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("J", "-1.0"));
  defaults.push_back(pair_t("lr", "5e-3"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("nthread", "1"));
  defaults.push_back(pair_t("ifprefix", "None"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int nInputs = parser.find<int>("ninput"),
    alpha = parser.find<int>("alpha"),
    nChains = parser.find<int>("ns"),
    nWarmup = parser.find<int>("nwarm"),
    nMonteCarloSteps = parser.find<int>("nms"),
    nIterations =  parser.find<int>("niter"),
    num_omp_threads = parser.find<int>("nthread"),
    version = parser.find<int>("ver");
  const double h = parser.find<double>("h"),
    J = parser.find<double>("J"),
    lr = parser.find<double>("lr");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string path = parser.find<>("path") + "/",
    nstr = std::to_string(nInputs),
    alphastr = std::to_string(alpha),
    vestr = std::to_string(version),
    ifprefix = parser.find<>("ifprefix");
  std::string hstr = std::to_string(h);
  hstr.erase(hstr.find_last_not_of('0') + 1, std::string::npos);
  hstr.erase(hstr.find_last_not_of('.') + 1, std::string::npos);
  // print info of the registered args
  parser.print(std::cout);

  // set number of threads for openmp
  omp_set_num_threads(num_omp_threads);

  FFNNSfSymm<double> machine(nInputs, alpha, nChains);

  // load parameters: w,a,b
  const std::string prefix = path + "FFNNSfSymm-CH-N" + nstr + "A" + alphastr + "H" + hstr + "V" + vestr;
  const std::string prefix0 = (ifprefix.compare("None")) ? path+ifprefix : prefix;
  machine.load(prefix0 + "-params.dat");

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(nIterations)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(nInputs)*
    static_cast<unsigned long>(nChains);

  // Transverse Field Ising Hamiltonian with 1D chain system
  TFIChain<AnsatzTraits<Ansatz::FFNNSfSymm, double> > Hsampler(machine, h, J, nBlocks, seed);
  const auto start = std::chrono::system_clock::now();

  Hsampler.warm_up(nWarmup);

  // imaginary time propagator
  StochasticReconfigurationCG<double> iTimePropagator(nChains, machine.get_nVariables());
  iTimePropagator.propagate(Hsampler, nIterations, nMonteCarloSteps, lr);

  const auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "# elapsed time: " << elapsed_seconds.count() << "(sec)" << std::endl;

  // save parameters: w,a,b
  machine.save(prefix + "-params.dat");

  return 0;
}
