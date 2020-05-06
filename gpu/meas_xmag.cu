// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include "measurements.cuh"
#include "../cpu/argparse.hpp"

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
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
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

  const uint32_t nInputs = parser.find<uint32_t>("ni"),
    nHiddens = parser.find<uint32_t>("nh"),
    nChains = parser.find<uint32_t>("ns"),
    nTrials = parser.find<uint32_t>("ntrials"),
    nWarmup = parser.find<uint32_t>("nwarm"),
    version = parser.find<uint32_t>("ver"),
    nMonteCarloSteps = parser.find<uint32_t>("nms"),
    deviceNumber = parser.find<uint32_t>("dev");
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
  machine.load(FNNDataType::W1, path + "/" + prefix + "Dw1.dat");
  machine.load(FNNDataType::W2, path + "/" + prefix + "Dw2.dat");
  machine.load(FNNDataType::B1, path + "/" + prefix + "Db1.dat");

  // block size for the block splitting scheme of parallel Monte-Carlo
  const uint64_t nBlocks = static_cast<uint64_t>(nTrials)*
    static_cast<uint64_t>(nMonteCarloSteps)*
    static_cast<uint64_t>(nInputs)*
    static_cast<uint64_t>(nChains);

  // measurements of the spontaneous magnetization with the given wave functions
  struct SamplerTraits { using AnsatzType = ComplexFNN<double>; using FloatType = double; };
  spinhalf::magnetization<double> outputs;
  spinhalf::MeasMagnetizationX<SamplerTraits> sampler(machine, nBlocks, seed);
  sampler.meas(nTrials, nWarmup, nMonteCarloSteps, outputs);

  std::cout << "# measurements outputs:" << std::endl
            << " -- m1: " << outputs.m1 << std::endl
            << " -- m2: " << outputs.m2 << std::endl;

  const std::string fileName = path + "/xmag-N" + nstr + ".dat";
  std::ofstream writer;
  if (!std::ifstream(fileName).is_open())
  {
    writer.open(fileName);
    writer << "#   h         m1        chi" << std::endl;
  }
  else
  {
    writer.open(fileName, std::fstream::app);
  }
  writer << std::setprecision(6);
  writer << std::setw(5) << h << " "
         << std::setw(10) << outputs.m1 << " "
	 << std::setw(10) << ((outputs.m2 - outputs.m1*outputs.m1)*nInputs) << std::endl;
  writer.close();
  return 0;
}
