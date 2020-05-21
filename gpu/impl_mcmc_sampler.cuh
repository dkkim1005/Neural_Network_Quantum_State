// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
BaseParallelSampler<DerivedParallelSampler, TraitsClass>::BaseParallelSampler(const uint32_t nMCUnitSteps,
  const uint32_t nChains, const unsigned long seedNumber, const unsigned long seedDistance):
  knMCUnitSteps(nMCUnitSteps),
  knChains(nChains),
  isNewStateAccepted_dev_(nChains, true),
  lnpsi0_dev_(nChains),
  lnpsi1_dev_(nChains),
  rngValues_dev_(nChains),
  rng_(seedNumber, seedDistance, nChains),
  kgpuBlockSize(1u+(nChains-1u)/NUM_THREADS_PER_BLOCK) {}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::warm_up(const uint32_t nMCSteps)
{
  // prepare an initial state
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->initialize_(PTR_FROM_THRUST(lnpsi0_dev_.data()));
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->accept_next_state_(PTR_FROM_THRUST(isNewStateAccepted_dev_.data()));
  // MCMC sampling for warming up
  this->do_mcmc_steps(nMCSteps);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::do_mcmc_steps(const uint32_t nMCSteps)
{
  // Markov chain MonteCarlo(MCMC) sampling with nskip iterations
  for (uint32_t n=0u; n<(nMCSteps*knMCUnitSteps); ++n)
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

namespace gpu_device
{
__device__ float Psi1OverPsi0Squared(const float Relnpsi0, const float Relnpsi1)
{
  const float delta[2] = {0.0f, 1.0f}, dlnpsi = Relnpsi1-Relnpsi0;
  return expf(2.0f*delta[(dlnpsi<0)]*dlnpsi);
}

__device__ double Psi1OverPsi0Squared(const double Relnpsi0, const double Relnpsi1)
{
  const double delta[2] = {0.0, 1.0}, dlnpsi = Relnpsi1-Relnpsi0;
  return exp(2.0*delta[(dlnpsi<0)]*dlnpsi);
}
}

namespace gpu_kernel
{
template <typename FloatType>
__global__ void Sampler__ParallelMetropolisUpdate__(
  const FloatType * rngValues,
  const uint32_t nChains,
  const thrust::complex<FloatType> * lnpsi1,
  thrust::complex<FloatType> * lnpsi0,
  bool * isNewStateAccepted)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType delta[2] = {0.0, 1.0};
  while (idx < nChains)
  {
    const FloatType ratio = gpu_device::Psi1OverPsi0Squared(lnpsi0[idx].real(), lnpsi1[idx].real());
    isNewStateAccepted[idx] = (rngValues[idx]<ratio);
    lnpsi0[idx] = lnpsi0[idx]+delta[isNewStateAccepted[idx]]*(lnpsi1[idx]-lnpsi0[idx]);
    idx += nstep;
  }
}
}
