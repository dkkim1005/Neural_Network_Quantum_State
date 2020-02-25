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
  using AnsatzType = typename Properties::AnsatzType;
  using FloatType = typename Properties::FloatType;
public:
  TFIChain(AnsatzType & machine, const FloatType h, const FloatType J, const unsigned long seedDistance,
    const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<FloatType> * trueGradients,
    const FloatType learningRate);
  AnsatzType & machine_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const FloatType kh, kJ, kzero, ktwo;
  std::vector<std::complex<FloatType> > diag_;
  std::vector<int> leftIdx_, rightIdx_;
};

// transverse field Ising model of the square lattice
template <typename Properties>
class TFISQ: public BaseParallelSampler<TFISQ, Properties>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFISQ, Properties)
  using AnsatzType = typename Properties::AnsatzType;
  using FloatType = typename Properties::FloatType;
public:
  TFISQ(AnsatzType & machine, const int L, const FloatType h, const FloatType J,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<FloatType> * trueGradients,
    const FloatType learningRate);
  AnsatzType & machine_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const int L_;
  const FloatType kh, kJ, kzero, ktwo;
  std::vector<std::complex<FloatType> > diag_;
  std::vector<int> lIdx_, rIdx_, uIdx_, dIdx_;
};

// transverse field Ising model of the triangular lattice
template <typename Properties>
class TFITRI: public BaseParallelSampler<TFITRI, Properties>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFITRI, Properties)
  using AnsatzType = typename Properties::AnsatzType;
  using FloatType = typename Properties::FloatType;
public:
  TFITRI(AnsatzType & machine, const int L, const FloatType h, const FloatType J,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<FloatType> * trueGradients,
    const FloatType learningRate);
  AnsatzType & machine_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const int L_;
  const FloatType kh, kJ, kzero, ktwo;
  std::vector<std::complex<FloatType> > diag_;
  std::vector<int> lIdx_, rIdx_, uIdx_, dIdx_, pIdx_, bIdx_;
};

#include "impl_hamiltonians.hpp"
