// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "common.cuh"
#include "neural_quantum_state.cuh"
#include "mcmc_sampler.cuh"
#include "hamiltonians.cuh"
#include "thrust_util.cuh"

template <typename TraitsClass>
class Sampler4SpinHalf : public BaseParallelSampler<Sampler4SpinHalf, TraitsClass>
{
  friend BaseParallelSampler<Sampler4SpinHalf, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  Sampler4SpinHalf(AnsatzType & psi, const unsigned long seedNumber, const unsigned long seedDistance);
  const thrust::complex<FloatType> * get_quantumStates() { return psi_.get_spinStates(); }
  int get_nInputs() const { return psi_.get_nInputs(); }
private:
  void initialize_(thrust::complex<FloatType> * lnpsi_dev);
  void sampling_(thrust::complex<FloatType> * lnpsi_dev);
  void accept_next_state_(bool * isNewStateAccepted_dev);
  AnsatzType & psi_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<>> list_;
};


/*---------------------------------------------------------------------------------------*
 * Renyi entropy S_2 = -log(Tr[\rho_A*\rho_A]), for \rho_A = Tr_B[|\psi><\psi|]
 *  |\psi> = \sum_{n_A,q_B} C(n_A,q_B) |n_A>(X)|q_B>, where C(n_A,q_B) = <n_A|(x)<q_B||\psi>
 *  => Tr[\rho_A*\rho_A] = \sum_{n_A,m_A,q_B,p_B} P(n_A,q_B)*P(m_A,p_B)*(\frac{C(n_A,p_B)*C(m_A,q_B)}{C(n_A,q_B)*C(m_A,p_B)})^*,
 *    where P(a_A,b_B) = |C(a_A,b_B)|^2 (probability of having the state |a_A>(x)|b_B>)
 * 
 *  Renyi entropy can be calculated with Monte-Carlo sampling of quanum states with
 *  a joint probability distribution P(n_A,q_B)*P(m_A,p_B).
 *---------------------------------------------------------------------------------------*/
template <typename TraitsClass>
class MeasRenyiEntropy
{
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  // smp1 : P(n_A,q_B), smp2 : P(m_A,p_B)
  MeasRenyiEntropy(Sampler4SpinHalf<TraitsClass> & smp1, Sampler4SpinHalf<TraitsClass> & smp2, AnsatzType & psi);
  // l : subregion length
  FloatType measure(const int l, const int nIterations, const int nMCSteps, const int nwarmup);
private:
  Sampler4SpinHalf<TraitsClass> & smp1_, & smp2_;
  AnsatzType & psi_;
  thrust::device_vector<thrust::complex<FloatType>> states3_dev_, states4_dev_, lnpsi3_dev_, lnpsi4_dev_;
  const thrust::complex<FloatType> kzero;
};

namespace gpu_kernel
{
template <typename FloatType>
__global__ void Renyi__SwapStates__(
  const int nChains,
  const int nInputs,
  const int l,
  thrust::complex<FloatType> * states1,
  thrust::complex<FloatType> * states2
);

template <typename FloatType>
__global__ void meas__GetRho2local__(
  const thrust::complex<FloatType> * lnpsi1,
  const thrust::complex<FloatType> * lnpsi2,
  const thrust::complex<FloatType> * lnpsi3,
  const thrust::complex<FloatType> * lnpsi4,
  const int nChains,
  thrust::complex<FloatType> * rho2local
);
} // namespace gpu_kernel


// calculating <\psi_1|\psi_2> with MCMC sampling
template <typename TraitsClass>
class MeasOverlapIntegral
{
  using AnsatzType = typename TraitsClass::AnsatzType;
  using AnsatzType2 = typename TraitsClass::AnsatzType2;
  using FloatType = typename TraitsClass::FloatType;
public:
  MeasOverlapIntegral(Sampler4SpinHalf<TraitsClass> & smp1, AnsatzType2 & m2);
  const thrust::complex<FloatType> get_overlapIntegral(const int nTrials,
    const int nwarms, const int nMCSteps = 1, const bool printStatics = false);
private:
  Sampler4SpinHalf<TraitsClass> & smp1_;
  AnsatzType2 & m2_;
  thrust::device_vector<thrust::complex<FloatType>> lnpsi2_dev_;
  const int knInputs, knChains, kgpuBlockSize;
  const thrust::complex<FloatType> kzero;
};


// calculating |<\psi_1|\psi_2>|^2 with MCMC sampling
template <typename TraitsClass>
class MeasFidelity
{
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  MeasFidelity(Sampler4SpinHalf<TraitsClass> & smp1, Sampler4SpinHalf<TraitsClass> & smp2, AnsatzType & psi1, AnsatzType & psi2);
  std::pair<FloatType, FloatType> measure(const int nTrials, const int nwarms, const int nMCSteps = 1);
private:
  Sampler4SpinHalf<TraitsClass> & smp1_, & smp2_;
  AnsatzType & psi1_, & psi2_;
  thrust::device_vector<thrust::complex<FloatType>> lnpsi3_dev_, lnpsi4_dev_;
  const int knInputs, knChains, kgpuBlockSize;
  const thrust::complex<FloatType> kzero;
};

namespace gpu_kernel
{
template <typename FloatType>
__global__ void meas__Psi2OverPsi0__(
  const thrust::complex<FloatType> * lnpsi0,
  const thrust::complex<FloatType> * lnpsi2,
  const int nChains,
  thrust::complex<FloatType> * psi2Overpsi0
);
} // namespace gpu_kernel


// z-z correlation function
template <typename TraitsClass>
class MeasSpinZSpinZCorrelation
{
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  explicit MeasSpinZSpinZCorrelation(Sampler4SpinHalf<TraitsClass> & smp);
  ~MeasSpinZSpinZCorrelation();
  void measure(const int nIterations, const int nMCSteps, const int nwarmup, FloatType * ss


#ifdef __KISTI_GPU__
  , const std::string logpath
#else

#endif


  );
private:
  Sampler4SpinHalf<TraitsClass> & smp_;
  thrust::device_vector<thrust::complex<FloatType>> ss_dev_; // spin-spin correlation in the spatial dimension
  const int knInputs, knChains;
  const thrust::complex<FloatType> kzero, kone;
  cublasHandle_t theCublasHandle_;
};


// x-x correlation function
template <typename TraitsClass>
class MeasSpinXSpinXCorrelation
{
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  MeasSpinXSpinXCorrelation(Sampler4SpinHalf<TraitsClass> & smp, AnsatzType & psi);
  ~MeasSpinXSpinXCorrelation();
  void measure(const int nIterations, const int nMCSteps, const int nwarmup, FloatType * ss, FloatType * s


#ifdef __KISTI_GPU__
, const std::string logpath
#endif


  );
private:
  Sampler4SpinHalf<TraitsClass> & smp_;
  AnsatzType & psi_;
  thrust::device_vector<thrust::complex<FloatType>> ss_dev_, s_dev_, tmplnpsi_dev_; // spin-spin correlation in the spatial dimension
  const thrust::device_vector<thrust::complex<FloatType>> kones;
  thrust::device_vector<thrust::pair<int, int>> spinPairFlipIdx_dev_;
  const int knInputs, knChains, kgpuBlockSize;
  const thrust::complex<FloatType> kzero, kone;
  cublasHandle_t theCublasHandle_;
};

namespace gpu_kernel
{
template <typename FloatType>
__global__ void meas__AccumPsi2OverPsi0__(
  const int nChains,
  const int nInputs,
  const thrust::complex<FloatType> * lnpsi_0,
  const thrust::complex<FloatType> * lnpsi_2,
  thrust::complex<FloatType> * ss);
} // end namespace gpu_kernel


// spontaneous magnetization m = \frac{1}{N}|\sum_{i}^{N} \sigma_i|
template <typename TraitsClass>
class MeasSpontaneousMagnetization
{
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  explicit MeasSpontaneousMagnetization(Sampler4SpinHalf<TraitsClass> & smp);
  ~MeasSpontaneousMagnetization();
  void measure(const int nIterations, const int nMCSteps, const int nwarmup, FloatType & m1, FloatType & m2, FloatType & m4);
private:
  Sampler4SpinHalf<TraitsClass> & smp_;
  cublasHandle_t theCublasHandle_;
  const thrust::complex<FloatType> kzero, kone, koneOverNinputs;
  const thrust::device_vector<thrust::complex<FloatType>> kones;
  thrust::device_vector<thrust::complex<FloatType>> tmpmag_dev_;
  thrust::device_vector<FloatType> mag_dev_;
};


// order parameter : \frac{1}{N}|\sum_{i}^{N} coeff_i*\sigma_i|
template <typename TraitsClass>
class MeasOrderParameter
{
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  explicit MeasOrderParameter(Sampler4SpinHalf<TraitsClass> & smp,
    const thrust::host_vector<thrust::complex<FloatType>> coeff_host);
  ~MeasOrderParameter();
  void measure(const int nIterations, const int nMCSteps, const int nwarmup, FloatType & m1, FloatType & m2, FloatType & m4);
private:
  Sampler4SpinHalf<TraitsClass> & smp_;
  cublasHandle_t theCublasHandle_;
  const thrust::complex<FloatType> kzero, kone, koneOverNinputs;
  thrust::device_vector<thrust::complex<FloatType>> coeff_dev_;
  thrust::device_vector<thrust::complex<FloatType>> tmpmag_dev_;
  thrust::device_vector<FloatType> mag_dev_;
};


namespace fermion
{
namespace jordanwigner
{
template <typename TraitsClass>
class Sampler4SpinHalf : public BaseParallelSampler<fermion::jordanwigner::Sampler4SpinHalf, TraitsClass>
{
  friend BaseParallelSampler<fermion::jordanwigner::Sampler4SpinHalf, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
  using LatticeTraits = typename TraitsClass::LatticeTraits;
public:
  Sampler4SpinHalf(AnsatzType & psi, const std::array<int, 2> & np,
    const unsigned long seedNumber, const unsigned long seedDistance);
  const thrust::complex<FloatType> * get_quantumStates() { return psi_.get_spinStates(); }
  int get_nInputs() const { return psi_.get_nInputs(); }
private:
  void initialize_(thrust::complex<FloatType> * lnpsi_dev);
  void sampling_(thrust::complex<FloatType> * lnpsi_dev);
  void accept_next_state_(bool * isNewStateAccepted_dev);
  AnsatzType & psi_;
  // # of particles for each flavor: up, down
  const std::array<int, 2> np_;
  const int knSites;
  kawasaki::NNSpinExchanger<LatticeTraits, FloatType> exchanger_;
  thrust::device_vector<thrust::pair<int, int> > spinPairIdx_dev_;
};


template <typename TraitsClass>
class MeasOPDM
{
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  MeasOPDM(fermion::jordanwigner::Sampler4SpinHalf<TraitsClass> & smp, AnsatzType & psi);
  // meas <b^+_{n+m}b_{n}> = <\psi|c^+_{n+m,up}c^+_{n+m,down}c_{n,down}c_{n,up}|\psi>
  std::complex<FloatType> measure(const int n, const int m, const int nIterations, const int nMCSteps, const int nwarmup);
private:
  fermion::jordanwigner::Sampler4SpinHalf<TraitsClass> & smp_;
  typename TraitsClass::AnsatzType & psi_;
  thrust::device_vector<thrust::complex<FloatType> > tmpspinStates_dev_, tmplnpsi_dev_, OPDM_dev_;
  const int knChains, knSites, kgpuBlockSize;
};
} // end namespace jordanwigner
} // end namespace fermion

namespace gpu_kernel
{
template <typename FloatType>
__global__ void OPDM__FlipSpins__(const int nChains, const int nSites,
  const int n, const int m, thrust::complex<FloatType> * spinStates);

template <typename FloatType>
__global__ void meas__OPDM__(const int nChains, const int nSites, const int n, const int m,
  const thrust::complex<FloatType> * spinStates, const thrust::complex<FloatType> * lnpsi0,
  const thrust::complex<FloatType> * lnpsi1, thrust::complex<FloatType> * OPDM);

template <typename FloatType>
__global__ void meas__OPDM__(const int nChains, const int nSites, const int n,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * OPDM);
} // end namespace gpu_kernel


#include "impl_meas.cuh"
