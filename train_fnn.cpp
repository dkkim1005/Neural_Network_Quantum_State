// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include <chrono>
#include "hamiltonians.hpp"
#include "optimizer.hpp"
#include "argparse.hpp"

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("ninput", "# of input nodes"));
  options.push_back(pair_t("nh", "# of hidden nodes"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of iterations to train FNN"));
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
  defaults.push_back(pair_t("lr", "1e-2"));
  defaults.push_back(pair_t("path", "."));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("nthread", "1"));
  defaults.push_back(pair_t("ifprefix", "None"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int nInputs = parser.find<int>("ninput"),
            nHiddens = parser.find<int>("nh"),
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
                    nistr = std::to_string(nInputs),
                    nhstr = std::to_string(nHiddens),
                    vestr = std::to_string(version),
                    ifprefix = parser.find<>("ifprefix");
  std::string hfstr = std::to_string(h);
  hfstr.erase(hfstr.find_last_not_of('0') + 1, std::string::npos);
  hfstr.erase(hfstr.find_last_not_of('.') + 1, std::string::npos);
  // print info of the registered args
  parser.print(std::cout);

  // set number of threads for openmp
  omp_set_num_threads(num_omp_threads);

  ComplexFNN<double> machine(nInputs, nHiddens, nChains);

  // load parameters
  const std::string prefix = path + "Ni" + nistr + "Nh" + nhstr + "Hf" + hfstr + "V" + vestr;
  const std::string prefix0 = (ifprefix.compare("None")) ? path+ifprefix : prefix;
  machine.load(FNNDataType::W1, prefix0 + "Dw1.dat");
  machine.load(FNNDataType::W2, prefix0 + "Dw2.dat");
  machine.load(FNNDataType::B1, prefix0 + "Db1.dat");

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(nIterations)*
                                static_cast<unsigned long>(nMonteCarloSteps)*
                                static_cast<unsigned long>(nInputs);

  // Transverse Field Ising Hamiltonian with 1D chain system
  TFIChain<AnsatzProperties<Ansatz::FNN, double> > sampler(machine, h, J, nBlocks, seed);
  const auto start = std::chrono::system_clock::now();

  sampler.warm_up(nWarmup);

  // imaginary time propagator
  StochasticReconfiguration<double, linearsolver::MINRESQLP> iTimePropagator(nChains, machine.get_nVariables());
  iTimePropagator.propagate(sampler, nIterations, nMonteCarloSteps, lr);

  const auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "# elapsed time: " << elapsed_seconds.count() << "(sec)" << std::endl;

  // save parameters
  machine.save(FNNDataType::W1, prefix + "Dw1.dat");
  machine.save(FNNDataType::W2, prefix + "Dw2.dat");
  machine.save(FNNDataType::B1, prefix + "Db1.dat");

  return 0;
}
