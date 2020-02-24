// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <fstream>
#include "measurements.hpp"
#include "hamiltonians.hpp"
#include "argparse.hpp"

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("pfile", "prefix of the first file name to load data of RBM"));
  options.push_back(pair_t("ntrials", "# of trials to compute overlap integral"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("nthread", "# of threads for openmp"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("nthread", "1"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int nChains = parser.find<int>("ns"),
            nTrials = parser.find<int>("ntrials"),
            nWarmup = parser.find<int>("nwarm"),
            nMonteCarloSteps = parser.find<int>("nms"),
            num_omp_threads = parser.find<int>("nthread");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string pfile = parser.find<>("pfile");
  parser.print(std::cout);

  const int nInputs = parsing_filename<int>(pfile, "Nv"),
            nHiddens = parsing_filename<int>(pfile, "Nh");

  // set number of threads for openmp
  omp_set_num_threads(num_omp_threads);

  ComplexRBM<double> machine(nInputs, nHiddens, nChains);

  // load parameters: w,a,b
  machine.load(RBMDataType::W, pfile + "Dw.dat");
  machine.load(RBMDataType::V, pfile + "Da.dat");
  machine.load(RBMDataType::H, pfile + "Db.dat");

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(nTrials)*
                                static_cast<unsigned long>(nMonteCarloSteps)*
                                static_cast<unsigned long>(nInputs);

  // measurements of the spontaneous magnetization with the given wave functions
  magnetization<double> outputs;
  MeasSpontaneousMagnetization<AnsatzProperties<Ansatz::RBM, double> > sampler(machine, nBlocks, seed);
  sampler.meas(nTrials, nWarmup, nMonteCarloSteps, outputs);

  std::cout << "# measurements outputs:" << std::endl
            << " -- m1: " << outputs.m1 << std::endl
            << " -- m2: " << outputs.m2 << std::endl
            << " -- m4: " << outputs.m4 << std::endl;

  return 0;
}
