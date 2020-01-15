// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "optimization.hpp"
#include "ComplexRBM.hpp"

template <typename FloatType>
class OneSideList
{
public:
  void set_item(const FloatType & item) { item_ = item; }
  void set_nextptr(OneSideList<FloatType> * nextPtr) { nextPtr_ = nextPtr; }
  OneSideList<FloatType> * next_ptr() const { return nextPtr_; }
  FloatType get_item() { return item_; }
private:
  FloatType item_;
  OneSideList<FloatType> * nextPtr_;
};


// transverse field Ising model for 1D chain.
template <typename FloatType>
class TFI_chain: public BaseParallelVMC<TFI_chain<FloatType>, FloatType>
{
  friend BaseParallelVMC<TFI_chain<FloatType>, FloatType>;
  using BaseParallelVMC<TFI_chain<FloatType>, FloatType>::lnpsi1_;
  using BaseParallelVMC<TFI_chain<FloatType>, FloatType>::lnpsi0_;
  typedef OneSideList<int> CircularLinkedList;
public:
  TFI_chain(ComplexRBM<FloatType> & machine, const FloatType h, const FloatType J, const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
  ComplexRBM<FloatType> & machine_;
  std::vector<CircularLinkedList> list_;
  CircularLinkedList * idxptr_;
  const FloatType kh, kJ, kzero, ktwo;
  std::vector<std::complex<FloatType> > diag_;
  std::vector<int> leftIdx_, rightIdx_;
};

#include "impl_hamiltonians.hpp"
