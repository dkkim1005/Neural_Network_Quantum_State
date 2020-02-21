// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <fstream>
#include "measurements.hpp"
#include "argparse.hpp"

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("pfile1", "prefix of the first file name to load data of RBM"));
  options.push_back(pair_t("pfile2", "prefix of the second file name to load data of RBM"));
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
            ntrials = parser.find<int>("ntrials"),
            nWarmup = parser.find<int>("nwarm"),
            nMonteCarloSteps = parser.find<int>("nms"),
            num_omp_threads = parser.find<int>("nthread");
  const unsigned long seed = parser.find<unsigned long>("seed");
  const std::string pfile1 = parser.find<>("pfile1"),
                    pfile2 = parser.find<>("pfile2");
  parser.print(std::cout);

  const int nInputs1 = parsing_filename<int>(pfile1, "Nv"),
            nHiddens1 = parsing_filename<int>(pfile1, "Nh");
  const int nInputs2 = parsing_filename<int>(pfile2, "Nv"),
            nHiddens2 = parsing_filename<int>(pfile2, "Nh");

  // set number of threads for openmp
  omp_set_num_threads(num_omp_threads);

  ComplexRBM<double> m1(nInputs1, nHiddens1, nChains), m2(nInputs2, nHiddens2, nChains);

  // load parameters: w,a,b
  m1.load(RBMDataType::W, pfile1 + "Dw.dat");
  m1.load(RBMDataType::V, pfile1 + "Da.dat");
  m1.load(RBMDataType::H, pfile1 + "Db.dat");
  m2.load(RBMDataType::W, pfile2 + "Dw.dat");
  m2.load(RBMDataType::V, pfile2 + "Da.dat");
  m2.load(RBMDataType::H, pfile2 + "Db.dat");

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(ntrials)*
                                static_cast<unsigned long>(nMonteCarloSteps)*
                                static_cast<unsigned long>(nInputs1);

  // measurements of the overlap integral for the given wave functions
  using ansatz_properties = AnsatzeProperties<Ansatz::RBM, Ansatz::RBM, double>;
  auto measPtr = std::make_unique<MeasOverlapIntegral<ansatz_properties> >(m1, m2, nBlocks, seed);
  const auto res1 = measPtr->get_overlapIntegral(ntrials, nWarmup, nMonteCarloSteps);
  std::cout << "# C_12*<\\psi_1|\\psi_2> : " << res1 << std::endl;
  measPtr.reset(new MeasOverlapIntegral<ansatz_properties>(m2, m1, nBlocks, seed));
  const auto res2 = measPtr->get_overlapIntegral(ntrials, nMonteCarloSteps);
  std::cout << "# C_21*<\\psi_2|\\psi_1> : " << res2 << std::endl;
  std::cout << "# |<\\psi_1|\\psi_2>|^2 : " << (res1*res2).real() << std::endl;

  // record measurements
  const double h1 = parsing_filename<double>(pfile1, "Hf"),
               h2 = parsing_filename<double>(pfile2, "Hf");
  const int ver1 = parsing_filename<int>(pfile1, "V"),
            ver2 = parsing_filename<int>(pfile2, "V");
  const std::string filename = "OI-Nv" + std::to_string(nInputs1) + ".dat";
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
