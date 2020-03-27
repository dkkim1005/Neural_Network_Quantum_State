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
  totalAcceptanceRatio_(0l),
  totalMeasurements_(0l),
  acceptanceRatio_(nChains)
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
      updateList_[k] = (randUniform_(randDev_[k]))<ratio_[k];
      lnpsi0_[k] = updateList_[k] ? lnpsi1_[k] : lnpsi0_[k];
      acceptanceRatio_[k] = acceptanceRatio_[k] + updateList_[k];
    }
    static_cast<DerivedParallelSampler<TraitsClass>*>(this)->accept_next_state(updateList_);
  }
  totalMeasurements_ = totalMeasurements_ + static_cast<unsigned long>(nMCSteps*knMCUnitSteps*knChains);
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

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
typename TraitsClass::FloatType BaseParallelSampler<DerivedParallelSampler, TraitsClass>::meas_acceptance_ratio()
{
  const FloatType acceptanceRatio = std::accumulate(acceptanceRatio_.begin(), acceptanceRatio_.end(), 0l)
    /static_cast<FloatType>(totalMeasurements_);
  std::fill(acceptanceRatio_.begin(), acceptanceRatio_.end(), 0l);
  totalMeasurements_ = 0l;
  return acceptanceRatio;
}


template <template<typename> class DerivedParallelTemperingSampler, typename TraitsClass>
BaseParallelTemperingSampler<DerivedParallelTemperingSampler, TraitsClass>::BaseParallelTemperingSampler(const int nSites,
  const int nChainsPerBeta, const int nBeta, const unsigned long seedDistance, const unsigned long seedNumber):
  knMCUnitSteps(nSites),
  knTotChains(nChainsPerBeta*nBeta),
  knChainsPerBeta(nChainsPerBeta),
  knBeta(nBeta),
  updateList_(nChainsPerBeta*nBeta),
  lnpsi0_(nChainsPerBeta*nBeta),
  lnpsi1_(nChainsPerBeta*nBeta),
  ratio_(nChainsPerBeta*nBeta),
  randDev_(nChainsPerBeta*nBeta),
  beta_(nBeta)
{
  if (nBeta%2 == 1)
    throw std::invalid_argument("nBeta must be even number.");
  // block splitting scheme for parallel Monte-Carlo
  for (int i=0; i<knTotChains; ++i)
  {
    randDev_[i].seed(seedNumber);
    randDev_[i].jump(2*seedDistance*i);
  }
  // set the domain of inverse temperature (1 --> 1/knBeta)
  for (int r=0; r<knBeta; ++r)
    beta_[r] = (knBeta-r)/static_cast<FloatType>(knBeta);
}

template <template<typename> class DerivedParallelTemperingSampler, typename TraitsClass>
void BaseParallelTemperingSampler<DerivedParallelTemperingSampler, TraitsClass>::warm_up(const int nMCSteps)
{
  // memorize an initial state
  static_cast<DerivedParallelTemperingSampler<TraitsClass>*>(this)->initialize(&lnpsi0_[0]);
  for (int k=0; k<knTotChains; ++k)
    updateList_[k] = true;
  static_cast<DerivedParallelTemperingSampler<TraitsClass>*>(this)->accept_next_state(updateList_);
  // MCMC sampling for warming up
  this->do_mcmc_steps(nMCSteps);
}

template <template<typename> class DerivedParallelTemperingSampler, typename TraitsClass>
void BaseParallelTemperingSampler<DerivedParallelTemperingSampler, TraitsClass>::do_mcmc_steps(const int nMCSteps)
{
  // replica exchange MCMC(parallel tempering) sampling for "nMCSteps" times
  // 1:1 ratio for 1 MC step(flip move) and state exchange(swap move)
  for (int nmc=0; nmc<nMCSteps; nmc++)
  {
    // local(flip) move
    for (int n=0; n<knMCUnitSteps; ++n)
    {
      static_cast<DerivedParallelTemperingSampler<TraitsClass>*>(this)->sampling(&lnpsi1_[0]);
      #pragma omp parallel for
      for (int r=0; r<knBeta; ++r)
        for (int k=0; k<knChainsPerBeta; ++k)
        {
          const int r_k = r*knChainsPerBeta+k;
          ratio_[r_k] = std::norm(std::exp(beta_[r]*(lnpsi1_[r_k]-lnpsi0_[r_k])));
          updateList_[r_k] = (randUniform_(randDev_[r_k]))<ratio_[r_k];
          lnpsi0_[r_k] = updateList_[r_k] ? lnpsi1_[r_k] : lnpsi0_[r_k];
        }
      static_cast<DerivedParallelTemperingSampler<TraitsClass>*>(this)->accept_next_state(updateList_);
    }
    // global(swap) move between r0=0,2,4,... and r1=1,3,5,... replicas, respectively
    #pragma omp parallel for
    for (int r0=0; r0<knBeta; r0+=2)
    {
      const FloatType dbeta_r0r1 = (beta_[r0]-beta_[r0+1]);
      for (int k=0; k<knChainsPerBeta; ++k)
      {
        const int r0_k = r0*knChainsPerBeta+k, r1_k = (r0+1)*knChainsPerBeta+k;
        ratio_[r0_k] = std::norm(std::exp(dbeta_r0r1*(-lnpsi0_[r0_k]+lnpsi0_[r1_k])));
        if (randUniform_(randDev_[r0_k])<ratio_[r0_k])
        {
          std::swap(lnpsi0_[r0_k], lnpsi0_[r1_k]);
          static_cast<DerivedParallelTemperingSampler<TraitsClass>*>(this)->swap_states(r0_k, r1_k);
        }
      }
    }
    // global(swap) move between r1=1,3,5,... and r2=2,4,6,... replicas, respectively
    #pragma omp parallel for
    for (int r1=1; r1<(knBeta-1); r1+=2)
    {
      const FloatType dbeta_r1r2 = (beta_[r1]-beta_[r1+1]);
      for (int k=0; k<knChainsPerBeta; ++k)
      {
        const int r1_k = r1*knChainsPerBeta+k, r2_k = (r1+1)*knChainsPerBeta+k;
        ratio_[r1_k] = std::norm(std::exp(dbeta_r1r2*(-lnpsi0_[r1_k]+lnpsi0_[r2_k])));
        if (randUniform_(randDev_[r1_k])<ratio_[r1_k])
        {
          std::swap(lnpsi0_[r1_k], lnpsi0_[r2_k]);
          static_cast<DerivedParallelTemperingSampler<TraitsClass>*>(this)->swap_states(r1_k, r2_k);
        }
      }
    }
  }
}

// Note that htilda range is [0, knChainsPerBeta)
template <template<typename> class DerivedParallelTemperingSampler, typename TraitsClass>
void BaseParallelTemperingSampler<DerivedParallelTemperingSampler, TraitsClass>::get_htilda(std::complex<FloatType> * htilda)
{
  static_cast<DerivedParallelTemperingSampler<TraitsClass>*>(this)->get_htilda(htilda);
}

// Note that lnpsiGradients range is [0, knChainsPerBeta*nVariables)
template <template<typename> class DerivedParallelTemperingSampler, typename TraitsClass>
void BaseParallelTemperingSampler<DerivedParallelTemperingSampler, TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  static_cast<DerivedParallelTemperingSampler<TraitsClass>*>(this)->get_lnpsiGradients(lnpsiGradients);
}

template <template<typename> class DerivedParallelTemperingSampler, typename TraitsClass>
void BaseParallelTemperingSampler<DerivedParallelTemperingSampler, TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  static_cast<DerivedParallelTemperingSampler<TraitsClass>*>(this)->evolve(trueGradients, learningRate);
}
