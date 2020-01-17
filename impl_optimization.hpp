// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <typename DerivedWFSampler, typename FloatType>
BaseParallelVMC<DerivedWFSampler, FloatType>::BaseParallelVMC(const int nMCUnitSteps, const int nChains,
  const unsigned long seedDistance, const unsigned long seedNumber):
  knMCUnitSteps(nMCUnitSteps),
  knChains(nChains),
  updateList_(nChains),
  lnpsi0_(nChains),
  lnpsi1_(nChains),
  ratio_(nChains),
  randDev_(nChains)
{
  // block splitting scheme for parallel Monte-Carlo
  for (int k=0; k<knChains; ++k)
  {
    randDev_[k].seed(seedNumber);
    randDev_[k].jump(2*seedDistance*k);
  }
}

template <typename DerivedWFSampler, typename FloatType>
void BaseParallelVMC<DerivedWFSampler, FloatType>::warm_up(const int nMCSteps)
{
  // memorize an initial state
  static_cast<DerivedWFSampler*>(this)->initialize(&lnpsi0_[0]);
  for (int k=0; k<knChains; ++k)
    updateList_[k] = true;
  static_cast<DerivedWFSampler*>(this)->accept_next_state(updateList_);
  // MCMC sampling for warming up
  this->do_mcmc_steps(nMCSteps);
}

template <typename DerivedWFSampler, typename FloatType>
void BaseParallelVMC<DerivedWFSampler, FloatType>::do_mcmc_steps(const int nMCSteps)
{
  // Markov chain MonteCarlo(MCMC) sampling with nskip iterations
  for (int n=0; n<(nMCSteps*knMCUnitSteps); ++n)
  {
    static_cast<DerivedWFSampler*>(this)->sampling(&lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChains; ++k)
      ratio_[k] = std::norm(std::exp(lnpsi1_[k]-lnpsi0_[k]));
    for (int k=0; k<knChains; ++k)
    {
      if (randUniform_(randDev_[k])<ratio_[k])
      {
        updateList_[k] = true;
        lnpsi0_[k] = lnpsi1_[k];
      }
      else
        updateList_[k] = false;
    }
    static_cast<DerivedWFSampler*>(this)->accept_next_state(updateList_);
  }
}

template <typename DerivedWFSampler, typename FloatType>
void BaseParallelVMC<DerivedWFSampler, FloatType>::get_htilda(std::complex<FloatType> * htilda)
{
  static_cast<DerivedWFSampler*>(this)->get_htilda(htilda);
}

template <typename DerivedWFSampler, typename FloatType>
void BaseParallelVMC<DerivedWFSampler, FloatType>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  static_cast<DerivedWFSampler*>(this)->get_lnpsiGradients(lnpsiGradients);
}

template <typename DerivedWFSampler, typename FloatType>
void BaseParallelVMC<DerivedWFSampler, FloatType>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  static_cast<DerivedWFSampler*>(this)->evolve(trueGradients, learningRate);
}


template <typename FloatType, template<typename> class LinearSolver>
StochasticReconfiguration<FloatType, LinearSolver>::StochasticReconfiguration(const int nChains, const int nVariables):
  htilda_(nChains),
  lnpsiGradients_(nChains*nVariables),
  kones(nChains, std::complex<FloatType>(1.0, 0.0)),
  koneOverNchains(std::complex<FloatType>(1.0/static_cast<FloatType>(nChains), 0.0)),
  kone(std::complex<FloatType>(1.0, 0.0)),
  kzero(std::complex<FloatType>(0.0, 0.0)),
  kminusOne(std::complex<FloatType>(-1.0, 0.0)),
  knChains(nChains),
  knVariables(nVariables),
  S_(nVariables*nVariables),
  aO_(nVariables),
  F_(nVariables),
  nIteration_(0),
  bp_(1.0),
  linSolver_(nVariables) {}

template <typename FloatType, template<typename> class LinearSolver>
FloatType StochasticReconfiguration<FloatType, LinearSolver>::schedular()
{
  bp_ *= kb;
  const FloatType lambda = klambda0*bp_;
  return ((lambda > klambMin) ? lambda : klambMin);
}
