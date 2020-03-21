// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "mcmc_sampler.hpp"
#include "neural_quantum_state.hpp"
#include "common.hpp"

namespace spinhalfsystem
{
// transverse field Ising model of the 1D chain
template <typename TraitsClass>
class TFIChain: public BaseParallelSampler<TFIChain, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFIChain, TraitsClass)
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
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
template <typename TraitsClass>
class TFISQ: public BaseParallelSampler<TFISQ, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFISQ, TraitsClass)
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
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
template <typename TraitsClass>
class TFITRI: public BaseParallelSampler<TFITRI, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFITRI, TraitsClass)
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
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


template <typename TraitsClass>
class TFICheckerBoard: public BaseParallelSampler<TFICheckerBoard, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFICheckerBoard, TraitsClass)
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  TFICheckerBoard(AnsatzType & machine, const int L, const FloatType h, const std::array<FloatType, 2> Jarr,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
  AnsatzType & machine_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const int kL;
  const FloatType kh, kJ1, kJ2, kzero, ktwo;
  std::vector<std::complex<FloatType> > diag_;
  std::vector<std::array<FloatType, 8> > Jmatrix_;
  std::vector<std::array<int, 8> > nnidx_;
};
} //  namespace spinhalfsystem

#include "impl_hamiltonians.hpp"
