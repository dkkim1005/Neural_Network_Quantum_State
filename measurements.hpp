// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <exception>
#include <numeric>
#include "mcmc_sampler.hpp"
#include "neural_quantum_state.hpp"
#include "hamiltonians.hpp"

template <Ansatz T1, Ansatz T2, typename Property>
struct AnsatzeProperties
{
  using AnsatzType1 = typename AnsatzType_<T1, Property>::Name;
  using AnsatzType2 = typename AnsatzType_<T2, Property>::Name;
  using FloatType = Property;
};

// calculating <\psi_1|\psi_2> with MCMC sampling
template <typename Properties>
class MeasOverlapIntegral : public BaseParallelSampler<MeasOverlapIntegral, Properties>
{
  USING_OF_BASE_PARALLEL_SAMPLER(MeasOverlapIntegral, Properties)
public:
  MeasOverlapIntegral(typename Properties::AnsatzType1 & m1, typename Properties::AnsatzType2 & m2,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  const std::complex<typename Properties::FloatType> get_overlapIntegral(const int nTrials, const int nwarms,
    const int nMCSteps = 1, const bool printStatics = false);
private:
  void initialize(std::complex<typename Properties::FloatType> * lnpsi);
  void sampling(std::complex<typename Properties::FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  typename Properties::AnsatzType1 & m1_;
  typename Properties::AnsatzType2 & m2_;
  std::vector<std::complex<typename Properties::FloatType> > lnpsi2_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
};


template <typename Properties>
MeasOverlapIntegral<Properties>::MeasOverlapIntegral(typename Properties::AnsatzType1 & m1,
  typename Properties::AnsatzType2 & m2, const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<MeasOverlapIntegral, Properties>(m1.get_nInputs(), m1.get_nChains(), seedDistance, seedNumber),
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

template <typename Properties>
const std::complex<typename Properties::FloatType> MeasOverlapIntegral<Properties>::get_overlapIntegral(const int nTrials,
  const int nwarms, const int nMCSteps, const bool printStatics)
{
  std::cout << "# Now we are in warming up..." << std::endl << std::flush;
  std::vector<std::complex<typename Properties::FloatType> > ovl(nTrials);
  this->warm_up(100);
  std::cout << "# Measuring overlap integrals... " << std::flush;
  for (int n=0; n<nTrials; ++n)
  {
    std::cout << (n+1) << " " << std::flush;
    this->do_mcmc_steps(nMCSteps);
    m2_.initialize(&lnpsi2_[0], m1_.get_spinStates());
    for (int i=0; i<lnpsi2_.size(); ++i)
      ovl[n] += std::exp(lnpsi2_[i]-lnpsi0_[i]);
    ovl[n] /= static_cast<typename Properties::FloatType>(lnpsi2_.size());
  }
  std::cout << "done." << std::endl;
  const std::complex<typename Properties::FloatType> ovlavg = std::accumulate(ovl.begin(), ovl.end(),
    std::complex<typename Properties::FloatType>(0,0))/static_cast<typename Properties::FloatType>(nTrials);
  if (printStatics)
  {
    typename Properties::FloatType realVar = 0, imagVar = 0;
    for (int n=0; n<nTrials; ++n)
    {
      realVar += std::pow(ovl[n].real()-ovlavg.real(), 2);
      imagVar += std::pow(ovl[n].imag()-ovlavg.imag(), 2);
    }
    realVar = std::sqrt(realVar/static_cast<typename Properties::FloatType>(nTrials-1));
    imagVar = std::sqrt(imagVar/static_cast<typename Properties::FloatType>(nTrials-1));
    std::cout << "# real part: " << ovlavg.real() << " +/- " << realVar << std::endl
              << "# imag part: " << ovlavg.imag() << " +/- " << imagVar << std::endl;
  }
  return ovlavg;
}

template <typename Properties>
void MeasOverlapIntegral<Properties>::initialize(std::complex<typename Properties::FloatType> * lnpsi)
{
  m1_.initialize(lnpsi);
}

template <typename Properties>
void MeasOverlapIntegral<Properties>::sampling(std::complex<typename Properties::FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  m1_.forward(idxptr_->get_item(), lnpsi);
}

template <typename Properties>
void MeasOverlapIntegral<Properties>::accept_next_state(const std::vector<bool> & updateList)
{
  m1_.spin_flip(updateList);
}
