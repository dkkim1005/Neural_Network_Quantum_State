// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "optimization.hpp"
#include "ComplexRBM.hpp"
#define USING_OF_BASE_PARALLEL_SAMPLER(DERIVED_PARALLEL_SAMPLER, FLOAT_TYPE)\
friend BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, FLOAT_TYPE>;\
using BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, FLOAT_TYPE>::lnpsi1_;\
using BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, FLOAT_TYPE>::lnpsi0_;


template <typename FloatType = int>
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


// transverse field Ising model of the 1D chain
template <typename FloatType>
class TFIChain: public BaseParallelSampler<TFIChain, FloatType>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFIChain, FloatType)
public:
  TFIChain(ComplexRBM<FloatType> & machine, const FloatType h, const FloatType J,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
  ComplexRBM<FloatType> & machine_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const FloatType kh, kJ, kzero, ktwo;
  std::vector<std::complex<FloatType> > diag_;
  std::vector<int> leftIdx_, rightIdx_;
};


// transverse field Ising model of the triangular lattice
template <typename FloatType>
class TFITRI: public BaseParallelSampler<TFITRI, FloatType>
{
  USING_OF_BASE_PARALLEL_SAMPLER(TFITRI, FloatType)
public:
  TFITRI(ComplexRBM<FloatType> & machine, const int L, const FloatType h, const FloatType J,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
  ComplexRBM<FloatType> & machine_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const int L_;
  const FloatType kh, kJ, kzero, ktwo;
  std::vector<std::complex<FloatType> > diag_;
  std::vector<int> lIdx_, rIdx_, uIdx_, dIdx_, pIdx_, bIdx_;
};

#include "impl_hamiltonians.hpp"
