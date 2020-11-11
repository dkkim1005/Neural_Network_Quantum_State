// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <exception>
#include <numeric>
#include <functional>
#include "mcmc_sampler.hpp"
#include "neural_quantum_state.hpp"
#include "common.hpp"

// calculating <\psi_1|\psi_2> with MCMC sampling
template <typename TraitsClass>
class MeasOverlapIntegral : public BaseParallelSampler<MeasOverlapIntegral, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(MeasOverlapIntegral, TraitsClass);
  using AnsatzType1 = typename TraitsClass::AnsatzType1;
  using AnsatzType2 = typename TraitsClass::AnsatzType2;
  using FloatType = typename TraitsClass::FloatType;
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


template <typename TraitsClass>
MeasOverlapIntegral<TraitsClass>::MeasOverlapIntegral(AnsatzType1 & m1, AnsatzType2 & m2,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<MeasOverlapIntegral, TraitsClass>(m1.get_nInputs(), m1.get_nChains(),
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

template <typename TraitsClass>
const std::complex<typename TraitsClass::FloatType> MeasOverlapIntegral<TraitsClass>::get_overlapIntegral(const int nTrials,
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

template <typename TraitsClass>
void MeasOverlapIntegral<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
{
  m1_.initialize(lnpsi);
}

template <typename TraitsClass>
void MeasOverlapIntegral<TraitsClass>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  m1_.forward(idxptr_->get_item(), lnpsi);
}

template <typename TraitsClass>
void MeasOverlapIntegral<TraitsClass>::accept_next_state(const std::vector<bool> & updateList)
{
  m1_.spin_flip(updateList);
}


template <template<typename> class Sampler, typename TraitsClass>
typename TraitsClass::FloatType meas_energy(BaseParallelSampler<Sampler, TraitsClass> & sampler,
  const int nTrials, const int nwarms, const int nMCSteps = 1)
{
  using FloatType = typename TraitsClass::FloatType;
  const int nChains = sampler.get_nChains();
  const std::complex<FloatType> zero(0.0, 0.0);
  FloatType etemp = zero.real();
  std::vector<std::complex<FloatType> > htilda(nChains);
  std::cout << "# warming up..." << std::endl << std::flush;
  sampler.warm_up(nwarms);
  std::cout << "# # of total measurements:" << nTrials*nChains << std::endl << std::flush;
  for (int n=0; n<nTrials; ++n)
  {
    std::cout << "# " << (n+1) << " / " << nTrials << std::endl << std::flush;
    sampler.do_mcmc_steps(nMCSteps);
    sampler.get_htilda(&htilda[0]);
    etemp = etemp + std::accumulate(htilda.begin(), htilda.end(), zero).real();
  }
  etemp = etemp/(nChains*nTrials);
  return etemp;
}


namespace spinhalf
{
template <typename FloatType>
struct magnetization { FloatType m1, m2, m4; };


template <typename TraitsClass>
class MeasSpontaneousMagnetization : public BaseParallelSampler<MeasSpontaneousMagnetization, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(MeasSpontaneousMagnetization, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
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

template <typename TraitsClass>
MeasSpontaneousMagnetization<TraitsClass>::MeasSpontaneousMagnetization(AnsatzType & machine,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<MeasSpontaneousMagnetization, TraitsClass>(machine.get_nInputs(),
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

template <typename TraitsClass>
void MeasSpontaneousMagnetization<TraitsClass>::meas(const int nTrials, const int nwarms,
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

template <typename TraitsClass>
void MeasSpontaneousMagnetization<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
}

template <typename TraitsClass>
void MeasSpontaneousMagnetization<TraitsClass>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename TraitsClass>
void MeasSpontaneousMagnetization<TraitsClass>::accept_next_state(const std::vector<bool> & updateList)
{
  machine_.spin_flip(updateList);
}


template <typename TraitsClass>
class MeasMagnetizationX : public BaseParallelSampler<MeasMagnetizationX, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(MeasMagnetizationX, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  MeasMagnetizationX(AnsatzType & machine, const unsigned long seedDistance, const unsigned long seedNumber = 0);
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

template <typename TraitsClass>
MeasMagnetizationX<TraitsClass>::MeasMagnetizationX(AnsatzType & machine,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<MeasMagnetizationX, TraitsClass>(machine.get_nInputs(),
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

template <typename TraitsClass>
void MeasMagnetizationX<TraitsClass>::meas(const int nTrials, const int nwarms,
  const int nMCSteps, magnetization<FloatType> & outputs)
{
  std::cout << "# Now we are in warming up...(" << nwarms << ")" << std::endl << std::flush;
  this->warm_up(nwarms);
  std::cout << "# # of total measurements:" << nTrials*knChains << std::endl << std::flush;
  FloatType mx1 = kzero, mx2 = kzero;
  std::vector<FloatType> mx1temp(knChains, kzero), mx2temp(knChains, kzero);
  const FloatType invNinputs = 1/static_cast<FloatType>(knInputs);
  const FloatType invNchains = 1/static_cast<FloatType>(knChains);
  const FloatType invNtrials = 1/static_cast<FloatType>(nTrials);
  const std::vector<bool> doSpinFlip(knChains, true);
  for (int n=0; n<nTrials; ++n)
  {
    std::cout << "# " << (n+1) << " / " << nTrials << std::endl << std::flush;
    this->do_mcmc_steps(nMCSteps);
    std::fill(mx1temp.begin(), mx1temp.end(), kzero);
    std::fill(mx2temp.begin(), mx2temp.end(), kzero);
    for (int i=0; i<knInputs; ++i)
    {
      machine_.forward(i, &lnpsi1_[0]);
      #pragma omp parallel for
      for (int k=0; k<knChains; ++k)
        mx1temp[k] += std::exp(lnpsi1_[k]-lnpsi0_[k]).real();
    }
    for (int i=0; i<knInputs; ++i)
    {
      machine_.spin_flip(doSpinFlip, i);
      for (int j=0; j<knInputs; ++j)
      {
        machine_.forward(j, &lnpsi1_[0]);
        #pragma omp parallel for
        for (int k=0; k<knChains; ++k)
          mx2temp[k] += std::exp(lnpsi1_[k]-lnpsi0_[k]).real();
      }
      machine_.spin_flip(doSpinFlip, i);
    }
    mx1 += std::accumulate(mx1temp.begin(), mx1temp.end(), kzero);
    mx2 += std::accumulate(mx2temp.begin(), mx2temp.end(), kzero);
  }
  mx1 *= (invNinputs*invNchains*invNtrials);
  mx2 *= (std::pow(invNinputs, 2)*invNchains*invNtrials);
  outputs.m1 = mx1;
  outputs.m2 = mx2;
}

template <typename TraitsClass>
void MeasMagnetizationX<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
}

template <typename TraitsClass>
void MeasMagnetizationX<TraitsClass>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename TraitsClass>
void MeasMagnetizationX<TraitsClass>::accept_next_state(const std::vector<bool> & updateList)
{
  machine_.spin_flip(updateList);
}


template <typename TraitsClass>
class MeasNeelOrder : public BaseParallelSampler<MeasNeelOrder, TraitsClass>
{
  USING_OF_BASE_PARALLEL_SAMPLER(MeasNeelOrder, TraitsClass);
  using AnsatzType = typename TraitsClass::AnsatzType;
  using FloatType = typename TraitsClass::FloatType;
public:
  MeasNeelOrder(AnsatzType & machine, const int L,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void meas(const int nTrials, const int nwarms, const int nMCSteps, magnetization<FloatType> & outputs);
private:
  void initialize(std::complex<FloatType> * lnpsi);
  void sampling(std::complex<FloatType> * lnpsi);
  void accept_next_state(const std::vector<bool> & updateList);
  AnsatzType & machine_;
  const std::complex<FloatType> * spinStates_;
  std::vector<OneWayLinkedIndex<> > list_;
  OneWayLinkedIndex<> * idxptr_;
  const int knInputs, knChains, kL;
  const FloatType kzero;
  std::vector<FloatType> coeff_;
};

template <typename TraitsClass>
MeasNeelOrder<TraitsClass>::MeasNeelOrder(AnsatzType & machine, const int L,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<MeasNeelOrder, TraitsClass>(machine.get_nInputs(),
    machine.get_nChains(), seedDistance, seedNumber),
  machine_(machine),
  list_(machine.get_nInputs()),
  knInputs(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kL(L),
  kzero(static_cast<FloatType>(0.0)),
  coeff_(L*L)
{
  if (knInputs != L*L)
    std::invalid_argument("machine.get_nInputs() != L*L");
  for (int i=0; i<knInputs; i++)
    list_[i].set_item(i);
  // black board(+1): (i+j)%2 == 0
  int idx0 = 0;
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((i+j)%2 == 1)
        continue;
      const int idx1 = (i*kL+j);
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // white board(-1): (i+j)%2 == 1
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((i+j)%2 == 0)
        continue;
      const int idx1 = i*kL+j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  list_[idx0].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
      coeff_[i*kL+j] = ((i+j)%2 == 0) ? 1 : -1;
}

template <typename TraitsClass>
void MeasNeelOrder<TraitsClass>::meas(const int nTrials, const int nwarms,
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
        mtemp[k] += (spinStates_[k*knInputs+i].real()*coeff_[i]);
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

template <typename TraitsClass>
void MeasNeelOrder<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
}

template <typename TraitsClass>
void MeasNeelOrder<TraitsClass>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename TraitsClass>
void MeasNeelOrder<TraitsClass>::accept_next_state(const std::vector<bool> & updateList)
{
  machine_.spin_flip(updateList);
}
} // namespace spinhalf
