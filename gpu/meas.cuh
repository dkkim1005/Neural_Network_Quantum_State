// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "common.cuh"
#include "neural_quantum_state.cuh"
#include "mcmc_sampler.cuh"
#include "hamiltonians.cuh"

template <typename TraitsClass>
class Sampler4SpinHalf : public BaseParallelSampler<Sampler4SpinHalf, TraitsClass>
{
  friend BaseParallelSampler<Sampler4SpinHalf, TraitsClass>;
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  Sampler4SpinHalf(AnsatzType & psi, const unsigned long seedNumber, const unsigned long seedDistance);
  const thrust::complex<FloatType> * get_quantumStates() { return psi_.get_spinStates(); }
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
  void measure(const int l, const int nIterations, const int nMCSteps, const int nwarmup);
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
__global__ void Renyi__GetRho2local__(
  const thrust::complex<FloatType> * lnpsi1,
  const thrust::complex<FloatType> * lnpsi2,
  const thrust::complex<FloatType> * lnpsi3,
  const thrust::complex<FloatType> * lnpsi4,
  const int nChains,
  thrust::complex<FloatType> * rho2local
);
} // namespace gpu_kernel

#include "impl_meas.cuh"
