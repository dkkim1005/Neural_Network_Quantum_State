// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "mcmc_sampler.cuh"
#include "neural_quantum_state.cuh"

template <typename FloatType = int> class OneWayLinkedIndex;

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
  TFISQ(AnsatzType & machine, const int L, const FloatType h, const FloatType J, const unsigned long seedNumber,
    const unsigned long seedDistance, const FloatType dropOutRate = 1, const std::string prefix = "./");
private:
  void get_htilda_(const thrust::complex<FloatType> * lnpsi0_dev,
    thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev);
  void get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev);
  void evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate);
  void initialize_(thrust::complex<FloatType> * lnpsi_dev);
  void sampling_(thrust::complex<FloatType> * lnpsi_dev);
  void accept_next_state_(bool * isNewStateAccepted_dev);
  void save_() const;
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<>> list_;
  thrust::device_vector<FloatType> diag_dev_;
  thrust::device_vector<int> nnidx_dev_;
  const int kL, knSites, knChains, kgpuBlockSize;
  const FloatType kh, kzero, ktwo;
  const thrust::device_vector<FloatType> kJmatrix_dev;
  const std::string kprefix;
  RandomBatchIndexing batchAllocater_;
};

// transverse field Ising model on the checker board lattice
template <typename TraitsClass>
class TFICheckerBoard: public BaseParallelSampler<TFICheckerBoard, TraitsClass>
{
  friend BaseParallelSampler<TFICheckerBoard, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  TFICheckerBoard(AnsatzType & machine, const int L, const FloatType h,
    const std::array<FloatType, 2> J1_J2, const bool isPeriodicBoundary, const unsigned long seedNumber,
    const unsigned long seedDistance, const FloatType dropOutRate = 1, const std::string prefix = "./");
private:
  void get_htilda_(const thrust::complex<FloatType> * lnpsi0_dev,
    thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev);
  void get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev);
  void evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate);
  void initialize_(thrust::complex<FloatType> * lnpsi_dev);
  void sampling_(thrust::complex<FloatType> * lnpsi);
  void accept_next_state_(bool * isNewStateAccepted_dev);
  void save_() const;
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<>> list_;
  thrust::device_vector<FloatType> diag_dev_, Jmatrix_dev_;
  thrust::device_vector<int> nnidx_dev_;
  const int kL, knSites, knChains, kgpuBlockSize;
  const FloatType kh, kJ1, kJ2, kzero, ktwo;
  const std::string kprefix;
  RandomBatchIndexing batchAllocater_;
};
} //  namespace spinhalf

namespace gpu_kernel
{
template <typename FloatType>
__global__ void TFI__GetDiagElem__(
  const thrust::complex<FloatType> * spinStates,
  const int nChains,
  const int nSites,
  const int * nnidx,
  const FloatType * Jmatrix,
  const int nnn,
  FloatType * diag
);

template <typename FloatType>
__global__ void TFI__UpdateDiagElem__(
  const thrust::complex<FloatType> * spinStates,
  const int nChains,
  const int nSites,
  const int * nnidx,
  const FloatType * Jmatrix,
  const int nnn,
  const bool * updateList,
  const int siteIdx,
  FloatType * diag
);

// htilda[k] += hfield*exp(lnpsi1[k] - lnpsi0[k]);
template <typename FloatType>
__global__ void TFI__GetOffDiagElem__(
  const int nChains,
  const FloatType hfield,
  const thrust::complex<FloatType> * lnpsi1,
  const thrust::complex<FloatType> * lnpsi0,
  thrust::complex<FloatType> * htilda
);
} // namespace gpu_kernel


// circular list structure
template <typename FloatType>
class OneWayLinkedIndex
{
public:
  void set_item(const FloatType & item) { item_ = item; }
  void set_nextptr(OneWayLinkedIndex * nextPtr) { nextPtr_ = nextPtr; }
  OneWayLinkedIndex * next_ptr() const { return nextPtr_; }
  FloatType get_item() { return item_; }
private:
  FloatType item_;
  OneWayLinkedIndex * nextPtr_;
};

#include "impl_hamiltonians.cuh"
