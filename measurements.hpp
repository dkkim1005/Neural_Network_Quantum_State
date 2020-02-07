// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <exception>
#include <numeric>
#include "mcmc_sampler.hpp"
#include "neural_quantum_state.hpp"
#include "hamiltonians.hpp"

// calculating <\psi_1|\psi_2> with MCMC sampling
template <typename FloatType>
class MeasOverlapIntegral : public BaseParallelSampler<MeasOverlapIntegral, FloatType>
{
  USING_OF_BASE_PARALLEL_SAMPLER(MeasOverlapIntegral, FloatType)
public:
  MeasOverlapIntegral(ComplexRBM<FloatType> & m1, ComplexRBM<FloatType> & m2,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  const std::complex<FloatType> get_overlapIntegral(const int nTrials, const int nwarms,
    const int nMCSteps = 1, const bool printStatics = false);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  ComplexRBM<FloatType> & m1_, & m2_;
  std::vector<std::complex<FloatType> > lnpsi2_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
};


template <typename FloatType>
MeasOverlapIntegral<FloatType>::MeasOverlapIntegral(ComplexRBM<FloatType> & m1, ComplexRBM<FloatType> & m2,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<MeasOverlapIntegral, FloatType>(m1.get_nInputs(), m1.get_nChains(), seedDistance, seedNumber),
  m1_(m1),
  m2_(m2),
  lnpsi2_(m1.get_nChains()),
  list_(m1.get_nInputs())
{
  try
  {
    if (m1.get_nInputs() != m2.get_nInputs())
      throw std::string("Error! Check the number of input nodes for each machine");
    if (m1.get_nChains() != m2.get_nChains())
      throw std::string("Error! Check the number of random number sequences for each machine");
  }
  catch (const std::string & errorMessage)
  {
    throw std::length_error(errorMessage);
  }
  // (checker board update) list_ : 1,3,5,...,2,4,6,...
  const int nSites = m1_.get_nInputs();
  for (int i=0; i<nSites; i++)
    list_[i].set_item(i);
  int idx = 0;
  for (int i=2; i<nSites; i+=2)
  {
    list_[idx].set_nextptr(&list_[i]);
    idx = i;
  }
  for (int i=1; i<nSites; i+=2)
  {
    list_[idx].set_nextptr(&list_[i]);
    idx = i;
  }
  list_[idx].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename FloatType>
const std::complex<FloatType> MeasOverlapIntegral<FloatType>::get_overlapIntegral(const int nTrials,
  const int nwarms, const int nMCSteps, const bool printStatics)
{
  std::cout << "# Now we are in warming up..." << std::endl << std::flush;
  std::vector<std::complex<FloatType> > ovl(nTrials);
  this->warm_up(100);
  std::cout << "# Measuring overlap integrals... " << std::flush;
  for (int n=0; n<nTrials; ++n)
  {
    std::cout << (n+1) << " " << std::flush;
    this->do_mcmc_steps(nMCSteps);
    m2_.initialize(&lnpsi2_[0], m1_.get_spinStates());
    for (int i=0; i<lnpsi2_.size(); ++i)
      ovl[n] += std::exp(lnpsi2_[i]-lnpsi0_[i]);
    ovl[n] /= static_cast<FloatType>(lnpsi2_.size());
  }
  std::cout << "done." << std::endl;
  const std::complex<FloatType> ovlavg = std::accumulate(ovl.begin(), ovl.end(),
    std::complex<FloatType>(0,0))/static_cast<FloatType>(nTrials);
  if (printStatics)
  {
    FloatType realVar = 0, imagVar = 0;
    for (int n=0; n<nTrials; ++n)
    {
      realVar += std::pow(ovl[n].real()-ovlavg.real(), 2);
      imagVar += std::pow(ovl[n].imag()-ovlavg.imag(), 2);
    }
    realVar = std::sqrt(realVar/static_cast<FloatType>(nTrials-1));
    imagVar = std::sqrt(imagVar/static_cast<FloatType>(nTrials-1));
    std::cout << "# real part: " << ovlavg.real() << " +/- " << realVar << std::endl
              << "# imag part: " << ovlavg.imag() << " +/- " << imagVar << std::endl;
  }
  return ovlavg;
}

template <typename FloatType>
void MeasOverlapIntegral<FloatType>::initialize(std::complex<FloatType> * lnpsi)
{
  m1_.initialize(lnpsi);
}

template <typename FloatType>
void MeasOverlapIntegral<FloatType>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  m1_.forward(idxptr_->get_item(), lnpsi);
}

template <typename FloatType>
void MeasOverlapIntegral<FloatType>::accept_next_state(const std::vector<bool> & updateList)
{
  m1_.spin_flip(updateList);
}
