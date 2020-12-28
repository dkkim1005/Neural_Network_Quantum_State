// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <fstream>
#include "../include/measurements.hpp"
#include "../include/hamiltonians.hpp"
#include "../include/argparse.hpp"

using namespace spinhalf;

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("ni", "# of input nodes"));
  options.push_back(pair_t("nh", "# of hidden nodes"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("ntrials", "# of trials to compute the spontaneous magnetization"));
  options.push_back(pair_t("h", "transverse-field strength"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("nthread", "# of threads for openmp"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("lattice", "lattice type(=CH,SQ,TRI,CB)"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("nthread", "1"));
  defaults.push_back(pair_t("path", "./"));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int nInputs = parser.find<int>("ni"),
            nHiddens = parser.find<int>("nh"),
            nChains = parser.find<int>("ns"),
            nTrials = parser.find<int>("ntrials"),
            nWarmup = parser.find<int>("nwarm"),
            version = parser.find<int>("ver"),
            nMonteCarloSteps = parser.find<int>("nms"),
            num_omp_threads = parser.find<int>("nthread");
  const double h = parser.find<double>("h");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string path = parser.find<>("path"),
                    nstr = std::to_string(nInputs),
                    nhstr = std::to_string(nHiddens),
                    vestr = std::to_string(version);
  std::string hfstr = std::to_string(h);
  hfstr.erase(hfstr.find_last_not_of('0') + 1, std::string::npos);
  hfstr.erase(hfstr.find_last_not_of('.') + 1, std::string::npos);
  const std::string lattice = parser.find<>("lattice") + "-",
                    prefix = lattice + "Ni" + nstr + "Nh" + nhstr + "Hf" + hfstr + "V" + vestr;

  parser.print(std::cout);

  // set number of threads for openmp
  omp_set_num_threads(num_omp_threads);

  FFNN<double> machine(nInputs, nHiddens, nChains);

  // load parameters
  machine.load(FFNNDataType::W1, path + "/" + prefix + "Dw1.dat");
  machine.load(FFNNDataType::W2, path + "/" + prefix + "Dw2.dat");
  machine.load(FFNNDataType::B1, path + "/" + prefix + "Db1.dat");

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(nTrials)*
                                static_cast<unsigned long>(nMonteCarloSteps)*
                                static_cast<unsigned long>(nInputs)*
                                static_cast<unsigned long>(nChains);

  // measurements of the spontaneous magnetization with the given wave functions
  magnetization<double> outputs;
  MeasMagnetizationX<AnsatzTraits<Ansatz::FFNN, double> > sampler(machine, nBlocks, seed);
  sampler.meas(nTrials, nWarmup, nMonteCarloSteps, outputs);

  std::cout << "# measurements outputs:" << std::endl
            << " -- m1: " << outputs.m1 << std::endl
            << " -- m2: " << outputs.m2 << std::endl;

  const std::string fileName = path + "/xmag-N" + nstr + ".dat";
  std::ofstream writer;
  if (!std::ifstream(fileName).is_open())
  {
    writer.open(fileName);
    writer << "#   h         m1        chi         U6" << std::endl;
  }
  else
  {
    writer.open(fileName, std::fstream::app);
  }
  writer << std::setprecision(6);
  writer << std::setw(5) << h << " "
         << std::setw(10) << outputs.m1 << " "
	 << std::setw(10) << ((outputs.m2 - outputs.m1*outputs.m1)*nInputs)
         << std::endl;
  writer.close();
  return 0;
}
