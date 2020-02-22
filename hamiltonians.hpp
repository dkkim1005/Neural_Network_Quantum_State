// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "mcmc_sampler.hpp"
#include "neural_quantum_state.hpp"

// List up all variational wave functions here...
enum class Ansatz {RBM, FNN};
template <Ansatz T, typename Property> struct AnsatzType_ {};
template <typename FloatType> struct AnsatzType_<Ansatz::RBM, FloatType> { using Name = ComplexRBM<FloatType>; };
template <typename FloatType> struct AnsatzType_<Ansatz::FNN, FloatType> { using Name = ComplexFNN<FloatType>; };
//
template <Ansatz T, typename Property>
struct AnsatzProperties
{
  using AnsatzType = typename AnsatzType_<T, Property>::Name;
  using FloatType = Property;
};

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
