// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "mcmc_sampler.cuh"
#include "neural_quantum_state.cuh"
#include "kawasaki_updater.cuh"

template <typename FloatType = int> class OneWayLinkedIndex;

namespace spinhalf
{
// transverse field Ising model on the 1D chain lattice
template <typename TraitsClass>
class TFIChain: public BaseParallelSampler<TFIChain, TraitsClass>
{
  friend BaseParallelSampler<TFIChain, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  TFIChain(AnsatzType & machine, const int L, const FloatType h, const FloatType J, const unsigned long seedNumber,
    const unsigned long seedDistance, const std::string prefix = "./");
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
  const int kL, knChains, kgpuBlockSize;
  const FloatType kh, kzero, ktwo;
  const thrust::device_vector<FloatType> kJmatrix_dev;
  const std::string kprefix;
};

// long-range interaction (J_{i,j} ~ 1/d(i,j)^{alpha}) Ising model on the 1D chain lattice
// Periodic boundary condition: circular positioning; d(i,j) = |i-j| if |i-j| < L/2, else L-|i-j|.
template <typename TraitsClass>
class LITFIChain: public BaseParallelSampler<LITFIChain, TraitsClass>
{
  friend BaseParallelSampler<LITFIChain, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  LITFIChain(AnsatzType & machine, const int L, const FloatType h,
    const FloatType J, const double alpha, const bool isPBC,
    const unsigned long seedNumber, const unsigned long seedDistance,
    const std::string prefix = "./");
  ~LITFIChain();
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
  thrust::device_vector<thrust::complex<FloatType> > Jmatrix_dev_, SJ_dev_;
  const int kL, knChains, kgpuBlockSize;
  const FloatType kh, kJ;
  const thrust::complex<FloatType> kzero, kone;
  const std::string kprefix;
  cublasHandle_t theCublasHandle_;
};

// transverse field Ising model on the square lattice
template <typename TraitsClass>
class TFISQ: public BaseParallelSampler<TFISQ, TraitsClass>
{
  friend BaseParallelSampler<TFISQ, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  TFISQ(AnsatzType & machine, const int L, const FloatType h, const FloatType J, const unsigned long seedNumber,
    const unsigned long seedDistance, const std::string prefix = "./");
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
    const unsigned long seedDistance, const std::string prefix = "./");
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
};
} //  namespace spinhalf

namespace fermion
{
namespace jordanwigner
{
// H = -t\sum_{ijs}(c^+_{i,s} c_{j,s}) + U\sum_{i}n_{i,up}n_{i,dw} + \sum_{i}V_{i}(c^+_{i,up}c_{i,up} + c^+_{i,dw}c_{i,dw})
template <typename TraitsClass>
class HubbardChain: public BaseParallelSampler<HubbardChain, TraitsClass>
{
  friend BaseParallelSampler<HubbardChain, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  HubbardChain(AnsatzType & machine, const FloatType U, const FloatType t,
    const std::vector<FloatType> & V, const std::array<int, 2> & np,
    const bool usePBC, const unsigned long seedNumber,
    const unsigned long seedDistance, const std::string prefix, const bool useSpinStates);
protected:
  void get_htilda_(const thrust::complex<FloatType> * lnpsi0_dev,
    thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev);
  void get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev);
  void evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate);
  void initialize_(thrust::complex<FloatType> * lnpsi_dev);
  void sampling_(thrust::complex<FloatType> * lnpsi_dev);
  void accept_next_state_(bool * isNewStateAccepted_dev);
  void save_() const;
  void load_spin_data_(thrust::host_vector<thrust::complex<FloatType>> & spinStates_host);
  void initialize_spins_randomly_(thrust::host_vector<thrust::complex<FloatType>> & spinStates_host);
  AnsatzType & machine_;
  const int knSites, knChains, kgpuBlockSize;
  const std::array<int, 2> np_;
  const bool kusePBC, kuseSpinStates;
  const FloatType kU, kt, kzero, ktwo;
  kawasaki::NNSpinExchanger<kawasaki::mChainLattice, FloatType> exchanger_;
  thrust::device_vector<thrust::pair<int, int> > spinPairIdx_dev_, tmpspinPairIdx_dev_;
  thrust::device_vector<FloatType> V_dev_;
  thrust::host_vector<thrust::complex<FloatType> > spinStates_host_;
  const std::string kprefix;
};
} // end namespace jordanwigner
} // end namespace fermion

namespace gpu_kernel
{
template <typename FloatType>
__global__ void TFI__GetDiagElem__(const thrust::complex<FloatType> * spinStates, const int nChains,
  const int nSites, const int * nnidx, const FloatType * Jmatrix, const int nnn, FloatType * diag);

template <typename FloatType>
__global__ void TFI__UpdateDiagElem__(const thrust::complex<FloatType> * spinStates, const int nChains,
  const int nSites, const int * nnidx, const FloatType * Jmatrix, const int nnn,
  const bool * updateList, const int siteIdx, FloatType * diag);

// htilda[k] += hfield*exp(lnpsi1[k] - lnpsi0[k]);
template <typename FloatType>
__global__ void TFI__GetOffDiagElem__(const int nChains, const FloatType hfield, const thrust::complex<FloatType> * lnpsi1,
  const thrust::complex<FloatType> * lnpsi0, thrust::complex<FloatType> * htilda);

template <typename FloatType>
__global__ void LITFI__GetDiagElem__(const thrust::complex<FloatType> * SJ, const thrust::complex<FloatType> * spinStates,
  const int nChains, const int nSites, thrust::complex<FloatType> * htilda);

template <typename FloatType>
__global__ void HubbardChain__AddedHoppingElem__(const int nChains, const int nSites,
  const thrust::complex<FloatType> * spinStates, const thrust::pair<int, int> * spinPairIdx,
  const thrust::complex<FloatType> * lnpsi0, const thrust::complex<FloatType> * lnpsi1, thrust::complex<FloatType> * htilda);

// flavor : 0(spin up) or 1(spin down)
template <typename FloatType>
__global__ void HubbardChain__AddedHoppingElemEdge__(const int flavor, const int nChains, const int nSites,
  const thrust::complex<FloatType> * spinStates, const thrust::pair<int, int> * spinPairIdx,
  const thrust::complex<FloatType> * lnpsi0, const thrust::complex<FloatType> * lnpsi1, thrust::complex<FloatType> * htilda);

template <typename FloatType>
__global__ void HubbardChain__AddedOnSiteInteraction__(const int nChains, const int nSites, const FloatType U,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * htilda);

template <typename FloatType>
__global__ void HubbardChain__AddedPotentialTrap__(const int nChains, const int nSites, const FloatType * V,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * htilda);
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
