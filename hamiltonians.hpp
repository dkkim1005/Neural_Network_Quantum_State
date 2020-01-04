#pragma once

#include "ComplexRBM.hpp"
#include <trng/yarn5.hpp>
#include <trng/uniform01_dist.hpp>
#include <trng/uniform_int_dist.hpp>

#define THIS_(TYPENAME) static_cast<TYPENAME*>(this)
#define CONST_THIS_(TYPENAME) static_cast<const TYPENAME*>(this)

/* 
 * Base class of importance sampling for wave functions: ln(\psi(x))
 *  - ratio = norm(ln(\psi(x1))-ln(\psi(x0)))
 *   where x1 is a candidate of the next state and x0 is a current state.
 */
template <typename DerivedWFSampler, typename float_t>
class BaseParallelVMC
{
public:
  BaseParallelVMC(const int nSites, const int nChains, const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void warm_up(const int nMCSteps = 100);
  void do_mcmc_steps(const int nMCSteps = 1);
  void get_htilda(std::complex<float_t> * htilda);
  void get_lnpsiGradients(std::complex<float_t> * lnpsiGradients);
  int get_nChains() const { return knChains; }
  void evolve(const std::complex<float_t> * trueGradients, const float_t learningRate);
private:
  const int knMCUnitSteps, knChains;
  std::vector<bool> updateList_;
  std::vector<float_t> ratio_;
  std::vector<trng::yarn5> randDev_;
  trng::uniform01_dist<float_t> randUniform_;
protected:
  std::vector<std::complex<float_t> > lnpsi1_, lnpsi0_;
};


template <typename float_t>
class OneSideList
{
public:
  void set_item(const float_t & item) { item_ = item; }
  void set_nextptr(OneSideList<float_t> * nextPtr) { nextPtr_ = nextPtr; }
  OneSideList<float_t> * next_ptr() const { return nextPtr_; }
  float_t get_item() { return item_; }
private:
  float_t item_;
  OneSideList<float_t> * nextPtr_;
};


// transverse field Ising model in 1D chain.
template <typename float_t>
class TFI_chain: public BaseParallelVMC<TFI_chain<float_t>, float_t>
{
  friend BaseParallelVMC<TFI_chain<float_t>, float_t>;
  using BaseParallelVMC<TFI_chain<float_t>, float_t>::lnpsi1_;
  using BaseParallelVMC<TFI_chain<float_t>, float_t>::lnpsi0_;
  typedef OneSideList<int> CircularLinkedList;
public:
  TFI_chain(ComplexRBM<float_t> & machine, const float_t h, const float_t J, const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void get_htilda(std::complex<float_t> * htilda);
  void get_lnpsiGradients(std::complex<float_t> * lnpsiGradients);
private:
  void initialize(std::complex<float_t> * lnpsi);
  void sampling(std::complex<float_t> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  void evolve(const std::complex<float_t> * trueGradients, const float_t learningRate);
  ComplexRBM<float_t> & machine_;
  std::vector<CircularLinkedList> list_;
  CircularLinkedList * idxptr_;
  const float_t kh, kJ, kzero, ktwo;
  std::vector<std::complex<float_t> > diag_;
  std::vector<int> leftIdx_, rightIdx_;
};

#include "impl_hamiltonians.hpp"
