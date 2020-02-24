// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "mcmc_sampler.hpp"
#include "neural_quantum_state.hpp"
#include "common.hpp"

// transverse field Ising model of the 1D chain
template <typename Properties>
class TFIChain: public BaseParallelSampler<TFIChain, Properties>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFIChain, Properties)
public:
  TFIChain(typename Properties::AnsatzType & machine, const typename Properties::FloatType h,
    const typename Properties::FloatType J, const unsigned long seedDistance,
    const unsigned long seedNumber = 0);
  void get_htilda(std::complex<typename Properties::FloatType> * htilda);
  void get_lnpsiGradients(std::complex<typename Properties::FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<typename Properties::FloatType> * lnpsi);
  void sampling(std::complex<typename Properties::FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<typename Properties::FloatType> * trueGradients,
    const typename Properties::FloatType learningRate);
  typename Properties::AnsatzType & machine_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const typename Properties::FloatType kh, kJ, kzero, ktwo;
  std::vector<std::complex<typename Properties::FloatType> > diag_;
  std::vector<int> leftIdx_, rightIdx_;
};

// transverse field Ising model of the square lattice
template <typename Properties>
class TFISQ: public BaseParallelSampler<TFISQ, Properties>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFISQ, Properties)
public:
  TFISQ(typename Properties::AnsatzType & machine, const int L,
    const typename Properties::FloatType h, const typename Properties::FloatType J,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<typename Properties::FloatType> * htilda);
  void get_lnpsiGradients(std::complex<typename Properties::FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<typename Properties::FloatType> * lnpsi);
  void sampling(std::complex<typename Properties::FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<typename Properties::FloatType> * trueGradients,
    const typename Properties::FloatType learningRate);
  typename Properties::AnsatzType & machine_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const int L_;
  const typename Properties::FloatType kh, kJ, kzero, ktwo;
  std::vector<std::complex<typename Properties::FloatType> > diag_;
  std::vector<int> lIdx_, rIdx_, uIdx_, dIdx_;
};

// transverse field Ising model of the triangular lattice
template <typename Properties>
class TFITRI: public BaseParallelSampler<TFITRI, Properties>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFITRI, Properties)
public:
  TFITRI(typename Properties::AnsatzType & machine, const int L,
    const typename Properties::FloatType h, const typename Properties::FloatType J,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<typename Properties::FloatType> * htilda);
  void get_lnpsiGradients(std::complex<typename Properties::FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<typename Properties::FloatType> * lnpsi);
  void sampling(std::complex<typename Properties::FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<typename Properties::FloatType> * trueGradients,
    const typename Properties::FloatType learningRate);
  typename Properties::AnsatzType & machine_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const int L_;
  const typename Properties::FloatType kh, kJ, kzero, ktwo;
  std::vector<std::complex<typename Properties::FloatType> > diag_;
  std::vector<int> lIdx_, rIdx_, uIdx_, dIdx_, pIdx_, bIdx_;
};

#include "impl_hamiltonians.hpp"
