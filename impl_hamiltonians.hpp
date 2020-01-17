// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <typename FloatType>
TFIChain<FloatType>::TFIChain(ComplexRBM<FloatType> & machine,
  const FloatType h, const FloatType J, const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelVMC<TFIChain<FloatType>, FloatType>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
  kh(h),
  kJ(J),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  leftIdx_(machine.get_nInputs()),
  rightIdx_(machine.get_nInputs())
{
  // (checker board update) list_ : 1,3,5,...,2,4,6,...
  const int nSites = machine.get_nInputs();
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
  // indexing of the nearest neighbor site(periodic boundary condition)
  for (int i=0; i<nSites; ++i)
  {
    leftIdx_[i] = ((i==0) ? nSites-1 : i-1);
    rightIdx_[i] = ((i==nSites-1) ? 0 : i+1);
  }
}

template <typename FloatType>
void TFIChain<FloatType>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<nChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<nSites-1; ++i)
      diag_[k] += spinPtr[k*nSites+i]*spinPtr[k*nSites+i+1];
    diag_[k] += spinPtr[k*nSites+nSites-1]*spinPtr[k*nSites];
  }
}

template <typename FloatType>
void TFIChain<FloatType>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename FloatType>
void TFIChain<FloatType>::accept_next_state(const std::vector<bool> & updateList)
{
  const int idx = idxptr_->get_item();
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  for (int k=0; k<nChains; ++k)
  {
    if (updateList[k])
      diag_[k] -= ktwo*spinPtr[k*nSites+idx]*(spinPtr[k*nSites+leftIdx_[idx]]+spinPtr[k*nSites+rightIdx_[idx]]);
  }
  machine_.spin_flip(updateList);
}

template <typename FloatType>
void TFIChain<FloatType>::get_htilda(std::complex<FloatType> * htilda)
{
  /*
     htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
      --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}
   */
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  for (int k=0; k<nChains; ++k)
    htilda[k] = kJ*diag_[k];
  for (int i=0; i<nSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<nChains; ++k)
      htilda[k] += kh*std::exp(lnpsi1_[k] - lnpsi0_[k]);
  }
}

template <typename FloatType>
void TFIChain<FloatType>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.backward(lnpsiGradients);
}

template <typename FloatType>
void TFIChain<FloatType>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_variables(trueGradients, learningRate);
}
