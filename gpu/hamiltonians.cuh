// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "mcmc_sampler.cuh"
#include "neural_quantum_state.cuh"

namespace spinhalf
{
// transverse field Ising model on the square lattice
template <typename TraitsClass>
class TFISQ: public BaseParallelSampler<TFISQ, TraitsClass>
{
  friend BaseParallelSampler<TFISQ, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  TFISQ(AnsatzType & machine, const int L, const FloatType h, const FloatType J,
    const unsigned long long seedNumber = 0ull, const FloatType dropOutRate = 1);
  void get_htilda(const thrust::complex<FloatType> * lnpsi0_dev,
    thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev);
  void get_lnpsiGradients(thrust::complex<FloatType> * lnpsiGradients_dev);
  void evolve(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate);
protected:
  void initialize(thrust::complex<FloatType> * lnpsi_dev);
  void sampling(thrust::complex<FloatType> * lnpsi_dev);
  void accept_next_state(bool * isNewStateAccepted_dev);
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<> > list_;
  thrust::device_vector<FloatType> diag_dev_;
  thrust::device_vector<int> nnidx_dev_;
  const int kL, knSites, knChains, kgpuBlockSize;
  const FloatType kh, kJ, kzero, ktwo;
  RandomBatchIndexing batchAllocater_;
};

/*
// transverse field Ising model on the checker board lattice
template <typename TraitsClass>
class TFICheckerBoard: public BaseParallelSampler<TFICheckerBoard, TraitsClass>
{
  friend BaseParallelSampler<TFICheckerBoard, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  TFICheckerBoard(AnsatzType & machine, const int L, const FloatType h,
    const std::array<FloatType, 2> J1_J2, const bool isPeriodicBoundary,
    const unsigned long long seedNumber = 0ull, const FloatType dropOutRate = 1);
  void get_htilda(const thrust::complex<FloatType> * lnpsi0_dev,
    thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev);
  void get_lnpsiGradients(thrust::complex<FloatType> * lnpsiGradients_dev);
  void evolve(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate);
private:
  void initialize(thrust::complex<FloatType> * lnpsi_dev);
  void sampling(thrust::complex<FloatType> * lnpsi);
  void accept_next_state(bool * isNewStateAccepted_dev);
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<> > list_;
  thrust::device_vector<FloatType> diag_dev_;
  thrust::device_vector<int> nnidx_dev_;
  std::vector<std::array<FloatType, 8> > Jmatrix_;
  const int kL, knSites, knChains, kgpuBlockSize;
  const FloatType kh, kJ1, kJ2, kzero, ktwo;
  RandomBatchIndexing<RandEngineType> batchAllocater_;
};
*/
} //  namespace spinhalf

namespace gpu_kernel
{
template <typename FloatType>
__global__ void TFI__InitDiag__(
  const thrust::complex<FloatType> * spinStates,
  const int nChains,
  const int nSites,
  const int * nnidx,
  const int nnn,
  FloatType * diag
);

template <typename FloatType>
__global__ void TFI__UpdateDiag__(
  const thrust::complex<FloatType> * spinStates,
  const int nChains,
  const int nSites,
  const int * nnidx,
  const int nnn,
  const bool * updateList,
  const int siteIdx,
  FloatType * diag
);

// htilda[k] += hfield*exp(lnpsi1[k] - lnpsi0[k]);
template <typename FloatType>
__global__ void TFI__GetHtildaStep1__(
  const int nChains,
  const FloatType hfield,
  const thrust::complex<FloatType> * lnpsi1,
  const thrust::complex<FloatType> * lnpsi0,
  thrust::complex<FloatType> * htilda
);

// htilda[k] += J*diag[k];
template <typename FloatType>
__global__ void TFI__GetHtildaStep2__(
  const int nChains,
  const FloatType J,
  const FloatType * diag,
  thrust::complex<FloatType> * htilda
);
} // namespace gpu_kernel

#include "impl_hamiltonians.cuh"
