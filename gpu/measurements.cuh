#pragma once

#include "mcmc_sampler.cuh"
#include "neural_quantum_state.cuh"
#include "common.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// calculating <\psi_1|\psi_2> with MCMC sampling
template <typename TraitsClass>
class MeasOverlapIntegral : public BaseParallelSampler<MeasOverlapIntegral, TraitsClass>
{
  friend BaseParallelSampler<MeasOverlapIntegral, TraitsClass>;
  using AnsatzType1 = typename TraitsClass::AnsatzType1;
  using AnsatzType2 = typename TraitsClass::AnsatzType2;
  using FloatType = typename TraitsClass::FloatType;
  using BaseParallelSampler<MeasOverlapIntegral, TraitsClass>::lnpsi0_dev_;
  using BaseParallelSampler<MeasOverlapIntegral, TraitsClass>::lnpsi1_dev_;
public:
  MeasOverlapIntegral(AnsatzType1 & m1, AnsatzType2 & m2, const uint64_t seedDistance,
    const uint64_t seedNumber = 0ul);
  const thrust::complex<FloatType> get_overlapIntegral(const uint32_t nTrials,
    const uint32_t nwarms, const uint32_t nMCSteps = 1u, const bool printStatics = false);
private:
  void initialize_(thrust::complex<FloatType> * lnpsi_dev);
  void sampling_(thrust::complex<FloatType> * lnpsi_dev);
  void accept_next_state_(const bool * isNewStateAccepted_dev);
  uint32_t idx_;
  AnsatzType1 & m1_;
  AnsatzType2 & m2_;
  thrust::device_vector<thrust::complex<FloatType>> lnpsi2_dev_;
  const uint32_t knInputs, knChains, kgpuBlockSize;
  const thrust::complex<FloatType> kzero;
};

template <typename SamplerType, typename FloatType>
FloatType meas_energy(SamplerType & sampler, const uint32_t nTrials, const uint32_t nwarms, const uint32_t nMCSteps = 1u)
{
  const uint32_t nChains = sampler.get_nChains();
  const thrust::complex<FloatType> zero(0, 0);
  FloatType groundEnergy = zero.real();
  thrust::device_vector<thrust::complex<FloatType>> htilda_dev(nChains);
  std::cout << "# warming up..." << std::endl << std::flush;
  sampler.warm_up(nwarms);
  std::cout << "# # of total measurements:" << nTrials*nChains << std::endl << std::flush;
  for (uint32_t n=0u; n<nTrials; ++n)
  {
    std::cout << "# " << (n+1) << " / " << nTrials << std::endl << std::flush;
    sampler.do_mcmc_steps(nMCSteps);
    sampler.get_htilda(PTR_FROM_THRUST(htilda_dev.data()));
    groundEnergy += (thrust::reduce(thrust::device, htilda_dev.begin(), htilda_dev.end(), zero)).real();
  }
  groundEnergy /= (nChains*nTrials);
  return groundEnergy;
}

namespace spinhalf
{
template <typename FloatType>
struct magnetization { FloatType m1, m2, m4; };

template <typename TraitsClass>
class MeasMagnetizationZ : public BaseParallelSampler<MeasMagnetizationZ, TraitsClass>
{
  friend BaseParallelSampler<MeasMagnetizationZ, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  MeasMagnetizationZ(AnsatzType & machine, const uint64_t seedDistance, const uint64_t seedNumber = 0ul);
  void meas(const uint32_t nTrials, const uint32_t nwarms, const uint32_t nMCSteps, magnetization<FloatType> & outputs);
private:
  void initialize_(thrust::complex<FloatType> * lnpsi_dev);
  void sampling_(thrust::complex<FloatType> * lnpsi_dev);
  void accept_next_state_(const bool * isNewStateAccepted_dev);
  uint32_t idx_;
  AnsatzType & machine_;
  const uint32_t knInputs, knChains, kgpuBlockSize;
  const FloatType kzero;
};

template <typename TraitsClass>
class MeasMagnetizationX : public BaseParallelSampler<MeasMagnetizationX, TraitsClass>
{
  friend BaseParallelSampler<MeasMagnetizationX, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
  using BaseParallelSampler<MeasMagnetizationX, TraitsClass>::lnpsi0_dev_;
  using BaseParallelSampler<MeasMagnetizationX, TraitsClass>::lnpsi1_dev_;
public:
  MeasMagnetizationX(AnsatzType & machine, const uint64_t seedDistance, const uint64_t seedNumber = 0ul);
  void meas(const uint32_t nTrials, const uint32_t nwarms, const uint32_t nMCSteps, magnetization<FloatType> & outputs);
private:
  void initialize_(thrust::complex<FloatType> * lnpsi_dev);
  void sampling_(thrust::complex<FloatType> * lnpsi_dev);
  void accept_next_state_(const bool * isNewStateAccepted_dev);
  uint32_t idx_;
  AnsatzType & machine_;
  const uint32_t knInputs, knChains, kgpuBlockSize;
  const FloatType kzero;
};

template <typename TraitsClass>
class MeasNeelOrder : public BaseParallelSampler<MeasNeelOrder, TraitsClass>
{
  friend BaseParallelSampler<MeasNeelOrder, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
  using BaseParallelSampler<MeasNeelOrder, TraitsClass>::lnpsi0_dev_;
  using BaseParallelSampler<MeasNeelOrder, TraitsClass>::lnpsi1_dev_;
public:
  MeasNeelOrder(AnsatzType & machine, const uint32_t L, const uint64_t seedDistance, const uint64_t seedNumber = 0ul);
  void meas(const uint32_t nTrials, const uint32_t nwarms, const uint32_t nMCSteps, magnetization<FloatType> & outputs);
private:
  void initialize_(thrust::complex<FloatType> * lnpsi_dev);
  void sampling_(thrust::complex<FloatType> * lnpsi_dev);
  void accept_next_state_(const bool * isNewStateAccepted_dev);
  AnsatzType & machine_;
  uint32_t idx_;
  const uint32_t knInputs, knChains, kL, kgpuBlockSize;
  const FloatType kzero;
  thrust::device_vector<FloatType> coeff_dev_;
};
} // spinhalf

namespace gpu_kernel
{
template <typename FloatType>
__global__ void meas__MeasAbsMagZ__(
  const thrust::complex<FloatType> * spinStates,
  const uint32_t nInputs,
  const uint32_t nChains,
  FloatType * m1,
  FloatType * m2,
  FloatType * m4
);

template <typename FloatType>
__global__ void meas__MeasAbsMagZ__(
  const thrust::complex<FloatType> * spinStates,
  const uint32_t nInputs,
  const uint32_t nChains,
  const FloatType * coeff,
  FloatType * m1,
  FloatType * m2,
  FloatType * m4
);

template <typename FloatType>
__global__ void meas__AccPsi1OverPsi0__(
  const thrust::complex<FloatType> * lnpsi0,
  const thrust::complex<FloatType> * lnpsi1,
  const uint32_t nChains,
  FloatType * mx
);

template <typename FloatType>
__global__ void meas__AccPsi2OverPsi0__(
  const thrust::complex<FloatType> * lnpsi0,
  const thrust::complex<FloatType> * lnpsi2,
  const uint32_t nChains,
  thrust::complex<FloatType> * psi2Overpsi0
);
} // gpu_kernel

#include "impl_measurements.cuh"