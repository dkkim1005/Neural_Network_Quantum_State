// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <fstream>
#include <cctype>
#include <algorithm>
#include "../include/meas.cuh"
#include "../../cpu/include/argparse.hpp"

inline bool is_number(const std::string s);
std::string query_for_string(const std::string str, const std::string query);

using namespace spinhalf;

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("file1", "file name to load data: 1"));
  options.push_back(pair_t("file2", "file name to load data: 2"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("ntrials", "# of trials to compute overlap integral"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("dev", "device number"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("path", "directory to load and save files"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "300"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("path", "."));
  // parser for arg list
  argsparse parser(argc, argv, options, defaults);
  const std::string path = parser.find<>("path") + "/",
    file1 = parser.find<>("file1"),
    file2 = parser.find<>("file2");
  const int nInputs = std::stoi(query_for_string(file1, "L")),
    nFilters1 = std::stoi(query_for_string(file1, "NF")),
    nFilters2 = std::stoi(query_for_string(file2, "NF")),
    nChains = parser.find<int>("ns"),
    ntrials = parser.find<int>("ntrials"),
    nWarmup = parser.find<int>("nwarm"),
    nMonteCarloSteps = parser.find<int>("nms"),
    deviceNumber = parser.find<int>("dev");
  const unsigned long seed = parser.find<unsigned long>("seed");

  if (nInputs != std::stoi(query_for_string(file2, "L")))
  {
    std::cerr << "# ERROR! The # of visible nodes (L) should be same. Check the file names to load data." << std::endl;
    return 1;
  }

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

  RBMTrSymm<double> m1(nInputs, nFilters1, nChains),
    m2(nInputs, nFilters2, nChains),
    psi1(nInputs, nFilters1, nChains),
    psi2(nInputs, nFilters2, nChains);

  // load parameters: w,a,b
  m1.load(path + parser.find<>("file1"));
  m2.load(path + parser.find<>("file2"));
  m1.copy_to(psi1);
  m2.copy_to(psi2);

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(ntrials)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(nInputs)*
    static_cast<unsigned long>(nChains);

  // measurements of the overlap integral for the given wave functions
  struct TRAITS { using AnsatzType = RBMTrSymm<double>; using FloatType = double; };

  Sampler4SpinHalf<TRAITS> smp1(m1, seed, nBlocks),
    smp2(m2, seed+987654321ul, nBlocks);

  MeasFidelity<TRAITS> fidelity(smp1, smp2, psi1, psi2);
  const auto res = fidelity.measure(ntrials, nWarmup, nMonteCarloSteps);
  std::cout << "# |<\\psi_1|\\psi_2>| : " << res.first << "  (+/-)  " << res.second << std::endl;

  // record measurements
  const std::string filename = path + "LICH-F-Ni" + std::to_string(nInputs) + ".dat";
  std::ofstream wfile;
  if(!std::ifstream(filename).is_open())
  {
    wfile.open(filename);
    wfile << "# alpha     nf1  theta1      v1     nf2  theta2      v2    seed      F       err"
          << std::endl;
  }
  else
    wfile.open(filename, std::ofstream::app);
  // format: alpha nf1 theta1 v1 nf2 theta2 v2 seed F err
  wfile << std::setprecision(7);
  wfile << std::setw(7) << query_for_string(file1, "A") << " "
        << std::setw(7) << nFilters1 << " "
        << std::setw(7) << query_for_string(file1, "T") << " "
        << std::setw(7) << query_for_string(file1, "V") << " "
        << std::setw(7) << nFilters2 << " "
        << std::setw(7) << query_for_string(file2, "T") << " "
        << std::setw(7) << query_for_string(file2, "V") << " "
        << std::setw(7) << seed << " "
        << std::setw(7) << res.first << " "
        << std::setw(7) << res.second << std::endl;
  wfile.close();
  return 0;
}


// REF: https://stackoverflow.com/questions/4654636/how-to-determine-if-a-string-is-a-number-with-c
// A minor edit had been done by D. Kim
inline bool is_number(const std::string s)
{
  return !s.empty() && std::find_if(s.begin(), s.end(),
    [](unsigned char c) {return !(std::isdigit(c) || c == '.');}) == s.end();
}

std::string query_for_string(const std::string str, const std::string query)
{
  const std::string emptyAnswer = "None";
  const int pos = str.find_last_of(query);
  if (pos == std::string::npos)
    return emptyAnswer;
  std::string answer;
  for (int i=pos+1; i<str.size(); ++i)
  {
    const std::string tmp = str.substr(i, 1);
    if (!is_number(tmp))
      break;
    answer += tmp;
  }
  return ((answer.size() != 0) ? answer : emptyAnswer);
}
