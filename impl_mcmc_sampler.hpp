// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
BaseParallelSampler<DerivedParallelSampler, TraitsClass>::BaseParallelSampler(const int nMCUnitSteps,
  const int nChains, const unsigned long seedDistance, const unsigned long seedNumber):
  knMCUnitSteps(nMCUnitSteps),
  knChains(nChains),
  updateList_(nChains),
  lnpsi0_(nChains),
  lnpsi1_(nChains),
  ratio_(nChains),
  randDev_(nChains),
  randUniform_(nChains)
{
  // block splitting scheme for parallel Monte-Carlo
  for (int k=0; k<knChains; ++k)
  {
    randDev_[k].seed(seedNumber);
    randDev_[k].jump(2*seedDistance*k);
  }
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::warm_up(const int nMCSteps)
{
  // memorize an initial state
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->initialize(&lnpsi0_[0]);
  for (int k=0; k<knChains; ++k)
    updateList_[k] = true;
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->accept_next_state(updateList_);
  // MCMC sampling for warming up
  this->do_mcmc_steps(nMCSteps);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::do_mcmc_steps(const int nMCSteps)
{
  // Markov chain MonteCarlo(MCMC) sampling with nskip iterations
  for (int n=0; n<(nMCSteps*knMCUnitSteps); ++n)
  {
    static_cast<DerivedParallelSampler<TraitsClass>*>(this)->sampling(&lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChains; ++k)
    {
      ratio_[k] = std::norm(std::exp(lnpsi1_[k]-lnpsi0_[k]));
      updateList_[k] = (randUniform_[k](randDev_[k]))<ratio_[k];
      lnpsi0_[k] = updateList_[k] ? lnpsi1_[k] : lnpsi0_[k];
    }
    static_cast<DerivedParallelSampler<TraitsClass>*>(this)->accept_next_state(updateList_);
  }
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::get_htilda(std::complex<FloatType> * htilda)
{
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->get_htilda(htilda);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->get_lnpsiGradients(lnpsiGradients);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->evolve(trueGradients, learningRate);
}
