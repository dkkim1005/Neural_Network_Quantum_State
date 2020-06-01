// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <fstream>
#include "meas.cuh"
#include "../cpu/argparse.hpp"

template <typename FloatType> std::string convert_from_float_to_string(const FloatType & num);
template <typename FloatType> std::string path_to_file(const argsparse & parser, const int i);
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) { return std::unique_ptr<T>(new T(std::forward<Args>(args)...)); }

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("Ni", "# of input nodes"));
  options.push_back(pair_t("Nh1", "# of hidden nodes 1"));
  options.push_back(pair_t("Nh2", "# of hidden nodes 2"));
  options.push_back(pair_t("ver1", "version 1"));
  options.push_back(pair_t("ver2", "version 2"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("h1", "transverse-field strength 1"));
  options.push_back(pair_t("h2", "transverse-field strength 2"));
  options.push_back(pair_t("ntrials", "# of trials to compute overlap integral"));
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
  defaults.push_back(pair_t("path", "."));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);
  const int nInputs = parser.find<int>("Ni"),
    nHiddens1 = parser.find<int>("Nh1"),
    nHiddens2 = parser.find<int>("Nh2"),
    nChains = parser.find<int>("ns"),
    ntrials = parser.find<int>("ntrials"),
    nWarmup = parser.find<int>("nwarm"),
    nMonteCarloSteps = parser.find<int>("nms"),
    deviceNumber = parser.find<int>("dev"),
    ver1 = parser.find<int>("ver1"),
    ver2 = parser.find<int>("ver2");
  const double h1 = parser.find<double>("h1"), h2 = parser.find<double>("h2");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string path1 = path_to_file<double>(parser, 1), path2 = path_to_file<double>(parser, 2);

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

  ComplexFNN<double> m1(nInputs, nHiddens1, nChains), m2(nInputs, nHiddens2, nChains);

  // load parameters: w,a,b
  m1.load(FNNDataType::W1, path1 + "Dw1.dat");
  m1.load(FNNDataType::W2, path1 + "Dw2.dat");
  m1.load(FNNDataType::B1, path1 + "Db1.dat");
  m2.load(FNNDataType::W1, path2 + "Dw1.dat");
  m2.load(FNNDataType::W2, path2 + "Dw2.dat");
  m2.load(FNNDataType::B1, path2 + "Db1.dat");

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(ntrials)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(nInputs)*
    static_cast<unsigned long>(nChains);

  // measurements of the overlap integral for the given wave functions
  struct TRAITS { using AnsatzType = ComplexFNN<double>;
                  using AnsatzType2 = ComplexFNN<double>;
                  using FloatType = double; };

  Sampler4SpinHalf<TRAITS> smp1(m1, seed, nBlocks), smp2(m2, seed+987654321ul, nBlocks);
  auto measPtr = make_unique<MeasOverlapIntegral<TRAITS>>(smp1, m2);
  const auto res1 = measPtr->get_overlapIntegral(ntrials, nWarmup, nMonteCarloSteps);
  std::cout << "# C_12*<\\psi_1|\\psi_2> : " << res1 << std::endl;
  measPtr.reset(new MeasOverlapIntegral<TRAITS>(smp2, m1));
  const auto res2 = measPtr->get_overlapIntegral(ntrials, nWarmup, nMonteCarloSteps);
  std::cout << "# C_21*<\\psi_2|\\psi_1> : " << res2 << std::endl;
  std::cout << "# |<\\psi_1|\\psi_2>|^2 : " << (res1*res2).real() << std::endl;

  // record measurements
  const std::string filename = parser.find<>("lattice")+ "-OI-Ni" + std::to_string(nInputs) + ".dat";
  std::ofstream wfile;
  if(!std::ifstream(filename).is_open())
  {
    wfile.open(filename);
    wfile << "#   nh1      h1      v1     nh2      h2      v2    seed       IO"
          << std::endl;
  }
  else
    wfile.open(filename, std::ofstream::app);
  // format: nh1 h1 v1 nh2 h2 v2 seed IO
  wfile << std::setprecision(7);
  wfile << std::setw(7) << nHiddens1 << " "
        << std::setw(7) << h1 << " "
        << std::setw(7) << ver1 << " "
        << std::setw(7) << nHiddens2 << " "
        << std::setw(7) << h2 << " "
        << std::setw(7) << ver2 << " "
        << std::setw(7) << seed << " "
        << std::setw(7) << (res1*res2).real() << std::endl;
  wfile.close();
  return 0;
}

template <typename FloatType>
std::string convert_from_float_to_string(const FloatType & num)
{
  std::string numstr = std::to_string(num);
  numstr.erase(numstr.find_last_not_of('0') + 1, std::string::npos);
  numstr.erase(numstr.find_last_not_of('.') + 1, std::string::npos);
  return numstr;
}

template <typename FloatType>
std::string path_to_file(const argsparse & parser, const int i)
{
  const std::string hQuery = "h"+std::to_string(i),
    NhQuery = "Nh"+std::to_string(i),
    vQuery = "ver"+std::to_string(i);
  const std::string filepath = parser.find<>("path") + "/"
    + parser.find<>("lattice") + "-Ni" + parser.find<>("Ni") + "Nh"
    + parser.find<>(NhQuery) + "Hf" + convert_from_float_to_string(parser.find<FloatType>(hQuery))
    + "V" + parser.find<>(vQuery);
  return filepath;
}
