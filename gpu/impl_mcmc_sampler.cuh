// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
BaseParallelSampler<DerivedParallelSampler, TraitsClass>::BaseParallelSampler(const int nMCUnitSteps,
  const int nChains, const unsigned long seedNumber, const unsigned long seedDistance):
  knMCUnitSteps(nMCUnitSteps),
  knChains(nChains),
  isNewStateAccepted_dev_(nChains, true),
  lnpsi0_dev_(nChains),
  lnpsi1_dev_(nChains),
  rngValues_dev_(nChains),
  rng_(seedNumber, seedDistance, nChains),
  kgpuBlockSize(1+(nChains-1)/NUM_THREADS_PER_BLOCK) {}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::warm_up(const int nMCSteps)
{
  // prepare an initial state
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->initialize_(PTR_FROM_THRUST(lnpsi0_dev_.data()));
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->accept_next_state_(PTR_FROM_THRUST(isNewStateAccepted_dev_.data()));
  // MCMC sampling for warming up
  this->do_mcmc_steps(nMCSteps);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::do_mcmc_steps(const int nMCSteps)
{
  // Markov chain MonteCarlo(MCMC) sampling with nskip iterations
  for (int n=0; n<(nMCSteps*knMCUnitSteps); ++n)
  {
    static_cast<DerivedParallelSampler<TraitsClass>*>(this)->sampling_(PTR_FROM_THRUST(lnpsi1_dev_.data()));
    rng_.get_uniformDist(PTR_FROM_THRUST(rngValues_dev_.data()));
    gpu_kernel::Sampler__ParallelMetropolisUpdate__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(rngValues_dev_.data()), knChains,
      PTR_FROM_THRUST(lnpsi1_dev_.data()), PTR_FROM_THRUST(lnpsi0_dev_.data()), PTR_FROM_THRUST(isNewStateAccepted_dev_.data()));
    static_cast<DerivedParallelSampler<TraitsClass>*>(this)->accept_next_state_(PTR_FROM_THRUST(isNewStateAccepted_dev_.data()));
  }
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::get_htilda(thrust::complex<FloatType> * htilda_dev)
{
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->get_htilda_(PTR_FROM_THRUST(lnpsi0_dev_.data()),
    PTR_FROM_THRUST(lnpsi1_dev_.data()), htilda_dev);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::get_lnpsiGradients(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->get_lnpsiGradients_(lnpsiGradients_dev);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::evolve(const thrust::complex<FloatType> * trueGradients_dev,
  const FloatType learningRate)
{
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->evolve_(trueGradients_dev, learningRate);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::save() const
{
  static_cast<const DerivedParallelSampler<TraitsClass>*>(this)->save_();
}


namespace gpu_kernel
{
template <typename FloatType>
__global__ void Sampler__ParallelMetropolisUpdate__(
  const FloatType * rngValues,
  const int nChains,
  const thrust::complex<FloatType> * lnpsi1,
  thrust::complex<FloatType> * lnpsi0,
  bool * isNewStateAccepted)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    const FloatType ratio = thrust::norm(thrust::exp(lnpsi1[idx]-lnpsi0[idx]));
    isNewStateAccepted[idx] = (rngValues[idx]<ratio);
    lnpsi0[idx] = (isNewStateAccepted[idx] ? lnpsi1[idx] : lnpsi0[idx]);
    idx += nstep;
  }
}
}
