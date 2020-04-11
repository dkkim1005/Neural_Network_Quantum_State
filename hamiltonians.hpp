// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <random>
#include "mcmc_sampler.hpp"
#include "neural_quantum_state.hpp"
#include "common.hpp"

namespace spinhalf
{
// transverse field Ising model on the 1D chain
template <typename TraitsClass>
class TFIChain: public BaseParallelSampler<TFIChain, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFIChain, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
  using RandEngineType = trng::yarn2;
public:
  TFIChain(AnsatzType & machine, const FloatType h, const FloatType J,
    const unsigned long seedDistance, const unsigned long seedNumber = 0,
    const FloatType dropOutRate = 1);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
protected:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<> > list_;
  std::vector<FloatType> diag_;
  std::vector<std::array<int, 2> > nnidx_;
  const int knSites, knChains;
  const FloatType kh, kJ, kzero, ktwo;
  RandomBatchIndexing<RandEngineType> batchAllocater_;
};


// transverse field Ising model on the square lattice
template <typename TraitsClass>
class TFISQ: public BaseParallelSampler<TFISQ, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFISQ, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
  using RandEngineType = trng::yarn2;
public:
  TFISQ(AnsatzType & machine, const int L, const FloatType h, const FloatType J,
    const unsigned long seedDistance, const unsigned long seedNumber = 0,
    const FloatType dropOutRate = 1);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
protected:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<> > list_;
  std::vector<FloatType> diag_;
  std::vector<std::array<int, 4> > nnidx_;
  const int kL, knSites, knChains;
  const FloatType kh, kJ, kzero, ktwo;
  RandomBatchIndexing<RandEngineType> batchAllocater_;
};


// transverse field Ising model on the triangular lattice
template <typename TraitsClass>
class TFITRI: public BaseParallelSampler<TFITRI, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFITRI, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
  using RandEngineType = trng::yarn2;
public:
  TFITRI(AnsatzType & machine, const int L, const FloatType h, const FloatType J,
    const unsigned long seedDistance, const unsigned long seedNumber = 0,
    const FloatType dropOutRate = 1);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<> > list_;
  std::vector<FloatType> diag_;
  std::vector<std::array<int, 6> > nnidx_;
  const int kL, knSites, knChains;
  const FloatType kh, kJ, kzero, ktwo;
  RandomBatchIndexing<RandEngineType> batchAllocater_;
};


// transverse field Ising model on the checker board lattice
template <typename TraitsClass>
class TFICheckerBoard: public BaseParallelSampler<TFICheckerBoard, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFICheckerBoard, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
  using RandEngineType = trng::yarn2;
public:
  TFICheckerBoard(AnsatzType & machine, const int L, const FloatType h,
    const std::array<FloatType, 2> J1_J2, const bool isBoundaryPeriodic,
    const unsigned long seedDistance, const unsigned long seedNumber = 0,
    const FloatType dropOutRate = 1);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<> > list_;
  std::vector<FloatType> diag_;
  std::vector<std::array<int, 8> > nnidx_;
  std::vector<std::array<FloatType, 8> > Jmatrix_;
  const int kL, knSites, knChains;
  const FloatType kh, kJ1, kJ2, kzero, ktwo;
  RandomBatchIndexing<RandEngineType> batchAllocater_;
};
} //  namespace spinhalf

namespace paralleltempering
{
namespace spinhalf
{
// transverse field Ising model on the 1D chain
template <typename TraitsClass>
class TFIChain: public BaseParallelTemperingSampler<TFIChain, TraitsClass>
{
  USING_OF_BASE_PARALLEL_TEMPERING_SAMPLER(TFIChain, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  TFIChain(AnsatzType & machine, const int nChainsPerBeta, const int nBeta, const FloatType h,
    const FloatType J, const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
  void swap_states(const int & k1, const int & k2);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<> > list_;
  std::vector<FloatType> diag_;
  std::vector<std::array<int, 2> > nnidx_;
  const int knSites, knTotChains, knChainsPerBeta, knBeta;
  const FloatType kh, kJ, kzero, ktwo;
};


// transverse field Ising model on the triangular lattice
template <typename TraitsClass>
class TFITRI: public BaseParallelTemperingSampler<TFITRI, TraitsClass>
{
  USING_OF_BASE_PARALLEL_TEMPERING_SAMPLER(TFITRI, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  TFITRI(AnsatzType & machine, const int L, const int nChainsPerBeta, const int nBeta,
    const FloatType h, const FloatType J, const unsigned long seedDistance,
    const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
  void swap_states(const int & k1, const int & k2);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<> > list_;
  std::vector<FloatType> diag_;
  std::vector<std::array<int, 6> > nnidx_;
  const int kL, knSites, knTotChains, knChainsPerBeta, knBeta;
  const FloatType kh, kJ, kzero, ktwo;
};


// transverse field Ising model on the checker board lattice
template <typename TraitsClass>
class TFICheckerBoard: public BaseParallelTemperingSampler<TFICheckerBoard, TraitsClass>
{
  USING_OF_BASE_PARALLEL_TEMPERING_SAMPLER(TFICheckerBoard, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  TFICheckerBoard(AnsatzType & machine, const int L, const int nChainsPerBeta, const int nBeta,
    const FloatType h, const std::array<FloatType, 2> J1_J2, const bool isBoundaryPeriodic,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
  void swap_states(const int & k1, const int & k2);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType & machine_;
  OneWayLinkedIndex<> * idxptr_;
  std::vector<OneWayLinkedIndex<> > list_;
  std::vector<FloatType> diag_;
  std::vector<std::array<int, 8> > nnidx_;
  std::vector<std::array<FloatType, 8> > Jmatrix_;
  const int kL, knSites, knTotChains, knChainsPerBeta, knBeta;
  const FloatType kh, kJ1, kJ2, kzero, ktwo;
};
} // namespace spinhalf
} // namespace paralleltempering

#include "impl_hamiltonians.hpp"
