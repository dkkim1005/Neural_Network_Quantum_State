#include "ComplexRBM.hpp"
#include "hamiltonians.hpp"
#include "optimization.hpp"
#include "argparse.hpp"

int main(int argc, char* argv[])
{
  std::vector<pair_t> options, defaults;
  // env; explanation of env
  options.push_back(pair_t("nv", "# of visible nodes"));
  options.push_back(pair_t("nh", "# of hidden nodes"));
  options.push_back(pair_t("ns", "# of spin samples for parallel Monte-Carlo"));
  options.push_back(pair_t("niter", "# of iterations to train RBM"));
  options.push_back(pair_t("h", "transverse-field strength"));
  options.push_back(pair_t("ver", "version"));
  options.push_back(pair_t("nwarm", "# of MCMC steps for warming-up"));
  options.push_back(pair_t("nms", "# of MCMC steps for sampling spins"));
  options.push_back(pair_t("J", "coupling constant"));
  options.push_back(pair_t("lr", "learning_rate"));
  options.push_back(pair_t("path", "directory to load and save files"));
  options.push_back(pair_t("seed", "seed of the parallel random number generator"));
  options.push_back(pair_t("nthread", "# of threads for openmp"));
  // env; default value
  defaults.push_back(pair_t("nwarm", "100"));
  defaults.push_back(pair_t("nms", "1"));
  defaults.push_back(pair_t("J", "-1.0"));
  defaults.push_back(pair_t("lr", "1e-2"));
  defaults.push_back(pair_t("path", "./"));
  defaults.push_back(pair_t("seed", "0"));
  defaults.push_back(pair_t("nthread", "1"));

  // parser for arg list
  argsparse parser(argc, argv, options, defaults);

  const int nInputs = parser.find<int>("nv"),
		    nHiddens = parser.find<int>("nh"),
		    nChains = parser.find<int>("ns"),
			nWarmup = parser.find<int>("nwarm"),
			nMonteCarloSteps = parser.find<int>("nms"),
			nIterations =  parser.find<int>("niter"),
	        num_omp_threads = parser.find<int>("nthread"),
            version = parser.find<int>("ver");
  const float h = parser.find<float>("h"),
		      J = parser.find<float>("J"),
			  lr = parser.find<float>("lr");
  const unsigned long seed = parser.find<unsigned long>("seed");

  const std::string path = parser.find<std::string>("path"),
					nvstr = std::to_string(nInputs),
					nhstr = std::to_string(nHiddens),
					vestr = std::to_string(version);
  std::string hfstr = std::to_string(h);
  hfstr.erase(hfstr.find_last_not_of('0') + 1, std::string::npos);
  hfstr.erase(hfstr.find_last_not_of('.') + 1, std::string::npos);
  const std::string prefix = path + "Nv" + nvstr + "Nh" + nhstr + "Hf" + hfstr + "V" + vestr;

  // print info of the registered args
  parser.print(std::cout);

  // set number of threads for openmp
  omp_set_num_threads(num_omp_threads);

  ComplexRBM<float> machine(nInputs, nHiddens, nChains);

  machine.load(RBMData_t::W, prefix + "Dw.dat");
  machine.load(RBMData_t::V, prefix + "Da.dat");
  machine.load(RBMData_t::H, prefix + "Db.dat");

  const unsigned long nJump = static_cast<unsigned long>(nIterations)*
                              static_cast<unsigned long>(nMonteCarloSteps)*
                              static_cast<unsigned long>(nInputs);
  TFI_chain<float> rbmWrapper(machine, h, J, nJump, seed);
  rbmWrapper.warm_up(nWarmup);

  // imaginary time propagator
  StochasticReconfiguration<float> iTimePropagator(nChains, machine.get_nVariables());

  iTimePropagator.propagate(rbmWrapper, nIterations, nMonteCarloSteps, lr);

  machine.save(RBMData_t::W, prefix + "Dw.dat");
  machine.save(RBMData_t::V, prefix + "Da.dat");
  machine.save(RBMData_t::H, prefix + "Db.dat");

  return 0;
}
