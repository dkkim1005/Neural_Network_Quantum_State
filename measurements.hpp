// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <exception>
#include <numeric>
#include <functional>
#include "mcmc_sampler.hpp"
#include "neural_quantum_state.hpp"
#include "common.hpp"

// calculating <\psi_1|\psi_2> with MCMC sampling
template <typename Properties>
class MeasOverlapIntegral : public BaseParallelSampler<MeasOverlapIntegral, Properties>
{
  USING_OF_BASE_PARALLEL_SAMPLER(MeasOverlapIntegral, Properties)
  using AnsatzType1 = typename Properties::AnsatzType1;
  using AnsatzType2 = typename Properties::AnsatzType2;
  using FloatType = typename Properties::FloatType;
public:
  MeasOverlapIntegral(AnsatzType1 & m1, AnsatzType2 & m2, const unsigned long seedDistance,
    const unsigned long seedNumber = 0);
  const std::complex<FloatType> get_overlapIntegral(const int nTrials, const int nwarms,
    const int nMCSteps = 1, const bool printStatics = false);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType1 & m1_;
  AnsatzType2 & m2_;
  std::vector<std::complex<FloatType> > lnpsi2_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
};


template <typename Properties>
MeasOverlapIntegral<Properties>::MeasOverlapIntegral(AnsatzType1 & m1, AnsatzType2 & m2,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<MeasOverlapIntegral, Properties>(m1.get_nInputs(), m1.get_nChains(),
    seedDistance, seedNumber),
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
  const int nSites = m1_.get_nInputs();
  for (int i=0; i<nSites; i++)
    list_[i].set_item(i);
  for (int i=0; i<nSites-1; i++)
    list_[i].set_nextptr(&list_[i+1]);
  list_[nSites-1].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename Properties>
const std::complex<typename Properties::FloatType> MeasOverlapIntegral<Properties>::get_overlapIntegral(const int nTrials,
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

template <typename Properties>
void MeasOverlapIntegral<Properties>::initialize(std::complex<FloatType> * lnpsi)
{
  m1_.initialize(lnpsi);
}

template <typename Properties>
void MeasOverlapIntegral<Properties>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  m1_.forward(idxptr_->get_item(), lnpsi);
}

template <typename Properties>
void MeasOverlapIntegral<Properties>::accept_next_state(const std::vector<bool> & updateList)
{
  m1_.spin_flip(updateList);
}


template <typename FloatType>
struct magnetization { FloatType m1, m2, m4; };


template <typename Properties>
class MeasSpontaneousMagnetization : public BaseParallelSampler<MeasSpontaneousMagnetization, Properties>
{
  USING_OF_BASE_PARALLEL_SAMPLER(MeasSpontaneousMagnetization, Properties)
  using AnsatzType = typename Properties::AnsatzType;
  using FloatType = typename Properties::FloatType;
public:
  MeasSpontaneousMagnetization(AnsatzType & machine, const unsigned long seedDistance,
    const unsigned long seedNumber = 0);
  void meas(const int nTrials, const int nwarms, const int nMCSteps, magnetization<FloatType> & outputs);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType & machine_;
  const std::complex<FloatType> * spinStates_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const int knInputs, knChains;
  const FloatType kzero;
};

template <typename Properties>
MeasSpontaneousMagnetization<Properties>::MeasSpontaneousMagnetization(AnsatzType & machine,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<MeasSpontaneousMagnetization, Properties>(machine.get_nInputs(),
    machine.get_nChains(), seedDistance, seedNumber),
  machine_(machine),
  list_(machine.get_nInputs()),
  knInputs(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kzero(static_cast<FloatType>(0.0))
{
  for (int i=0; i<knInputs; i++)
    list_[i].set_item(i);
  for (int i=0; i<knInputs-1; i++)
    list_[i].set_nextptr(&list_[i+1]);
  list_[knInputs-1].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename Properties>
void MeasSpontaneousMagnetization<Properties>::meas(const int nTrials, const int nwarms,
  const int nMCSteps, magnetization<FloatType> & outputs)
{
  std::cout << "# Now we are in warming up...(" << nwarms << ")" << std::endl << std::flush;
  this->warm_up(nwarms);
  std::cout << "# # of total measurements:" << nTrials*knChains << std::endl << std::flush;
  const auto Lambda2Sum = [](FloatType & a,
    FloatType & b)->FloatType {return a+(b*b);};
  const auto Lambda4Sum = [](FloatType & a,
    FloatType & b)->FloatType {return a+(b*b*b*b);};
  std::vector<FloatType> m1arr(nTrials, kzero), m2arr(nTrials, kzero),
    m4arr(nTrials, kzero), mtemp(knChains, kzero);
  const FloatType invNinputs = 1/static_cast<FloatType>(knInputs);
  const FloatType invNchains = 1/static_cast<FloatType>(knChains);
  const FloatType invNtrials = 1/static_cast<FloatType>(nTrials);
  for (int n=0; n<nTrials; ++n)
  {
    std::cout << "# " << (n+1) << " / " << nTrials << std::endl << std::flush;
    this->do_mcmc_steps(nMCSteps);
    spinStates_ = machine_.get_spinStates();
    std::fill(mtemp.begin(), mtemp.end(), kzero);
    #pragma omp parallel for
    for (int k=0; k<knChains; ++k)
    {
      for (int i=0; i<knInputs; ++i)
        mtemp[k] += spinStates_[k*knInputs+i].real();
      mtemp[k] = std::abs(mtemp[k])*invNinputs;
    }
    m1arr[n] = std::accumulate(mtemp.begin(), mtemp.end(), kzero)*invNchains;
    m2arr[n] = std::accumulate(mtemp.begin(), mtemp.end(), kzero, Lambda2Sum)*invNchains;
    m4arr[n] = std::accumulate(mtemp.begin(), mtemp.end(), kzero, Lambda4Sum)*invNchains;
  }
  outputs.m1 = std::accumulate(m1arr.begin(), m1arr.end(), kzero)*invNtrials;
  outputs.m2 = std::accumulate(m2arr.begin(), m2arr.end(), kzero)*invNtrials;
  outputs.m4 = std::accumulate(m4arr.begin(), m4arr.end(), kzero)*invNtrials;
}

template <typename Properties>
void MeasSpontaneousMagnetization<Properties>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
}

template <typename Properties>
void MeasSpontaneousMagnetization<Properties>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename Properties>
void MeasSpontaneousMagnetization<Properties>::accept_next_state(const std::vector<bool> & updateList)
{
  machine_.spin_flip(updateList);
}
