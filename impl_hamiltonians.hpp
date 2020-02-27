// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

namespace spinhalfsystem
{
template <typename TraitsClass>
TFIChain<TraitsClass>::TFIChain(AnsatzType & machine, const FloatType h,
  const FloatType J, const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<TFIChain, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
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

template <typename TraitsClass>
void TFIChain<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
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

template <typename TraitsClass>
void TFIChain<TraitsClass>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::accept_next_state(const std::vector<bool> & updateList)
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

template <typename TraitsClass>
void TFIChain<TraitsClass>::get_htilda(std::complex<FloatType> * htilda)
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

template <typename TraitsClass>
void TFIChain<TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.backward(lnpsiGradients);
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_variables(trueGradients, learningRate);
}


template <typename TraitsClass>
TFISQ<TraitsClass>::TFISQ(AnsatzType & machine, const int L,
  const FloatType h, const FloatType J,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<TFISQ, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
  L_(L),
  kh(h),
  kJ(J),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  lIdx_(machine.get_nInputs()),
  rIdx_(machine.get_nInputs()),
  uIdx_(machine.get_nInputs()),
  dIdx_(machine.get_nInputs())
{
  if (L_*L_ != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L*L!");
  for (int i=0; i<L_; ++i)
    for (int j=0; j<L_; ++j)
    {
      lIdx_[i*L_+j] = (j!=0) ? L_*i+j-1 : L_*i+L_-1;
      rIdx_[i*L_+j] = (j!=L_-1) ? L_*i+j+1 : L_*i;
      uIdx_[i*L_+j] = (i!=0) ? L_*(i-1)+j : L_*(L_-1)+j;
      dIdx_[i*L_+j] = (i!=L_-1) ? L_*(i+1)+j : j;
    }

  // Checkerboard link(To implement the MCMC update rule)
  for (int i=0; i<L*L; ++i)
    list_[i].set_item(i);
  // black board: (i+j)%2 == 0
  int idx0 = 0;
  for (int i=0; i<L_; ++i)
    for (int j=0; j<L_; ++j)
    {
      if ((i+j)%2 != 0)
        continue;
      const int idx1 = i*L_ + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // white board: (i+j)%2 == 1
  for (int i=0; i<L_; ++i)
    for (int j=0; j<L_; ++j)
    {
      if ((i+j)%2 != 1)
        continue;
      const int idx1 = i*L_ + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  list_[idx0].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<nChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<L_; ++i)
      for (int j=0; j<L_; ++j)
      {
        const int idx = i*L_+j;
        diag_[k] += spinPtr[k*nSites+idx]*
                   (spinPtr[k*nSites+lIdx_[idx]]+spinPtr[k*nSites+rIdx_[idx]]+
                    spinPtr[k*nSites+uIdx_[idx]]+spinPtr[k*nSites+dIdx_[idx]]);
      }
    diag_[k] *= 0.5;
  }
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::accept_next_state(const std::vector<bool> & updateList)
{
  const int idx = idxptr_->get_item();
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  for (int k=0; k<nChains; ++k)
  {
    if (updateList[k])
      diag_[k] -= ktwo*spinPtr[k*nSites+idx]*
                   (spinPtr[k*nSites+lIdx_[idx]] +
                    spinPtr[k*nSites+rIdx_[idx]] +
                    spinPtr[k*nSites+uIdx_[idx]] +
                    spinPtr[k*nSites+dIdx_[idx]]);
  }
  machine_.spin_flip(updateList);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::get_htilda(std::complex<FloatType> * htilda)
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

template <typename TraitsClass>
void TFISQ<TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.backward(lnpsiGradients);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_variables(trueGradients, learningRate);
}


template <typename TraitsClass>
TFITRI<TraitsClass>::TFITRI(AnsatzType & machine, const int L,
  const FloatType h, const FloatType J,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<TFITRI, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
  L_(L),
  kh(h),
  kJ(J),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  lIdx_(machine.get_nInputs()),
  rIdx_(machine.get_nInputs()),
  uIdx_(machine.get_nInputs()),
  dIdx_(machine.get_nInputs()),
  pIdx_(machine.get_nInputs()),
  bIdx_(machine.get_nInputs())
{
  if (L_*L_ != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L*L!");
  for (int i=1; i<L_-1; ++i)
    for (int j=1; j<L_-1; ++j)
    {
      lIdx_[i*L_+j] = L_*(i-1)+j-1;
      rIdx_[i*L_+j] = L_*(i-1)+j;
      uIdx_[i*L_+j] = L_*i+j-1;
      dIdx_[i*L_+j] = L_*i+j+1;
      pIdx_[i*L_+j] = L_*(i+1)+j;
      bIdx_[i*L_+j] = L_*(i+1)+j+1;
    }
  // case i=0, j=0
  lIdx_[0] = L_*L_-1, rIdx_[0] = L_*(L_-1), uIdx_[0] = L_-1,
  dIdx_[0] = 1, pIdx_[0] = L_, bIdx_[0] = L_+1;
  // case i=0, j=L-1
  lIdx_[L-1] = L_*L_-2, rIdx_[L-1] = L_*L_-1, uIdx_[L-1] = L_-2,
  dIdx_[L-1] = 0, pIdx_[L-1] = L_+L_-1, bIdx_[L-1] = L_;
  // case i=L-1, j=0
  lIdx_[(L-1)*L_] = L_*(L_-2)+L_-1, rIdx_[(L-1)*L_] = L_*(L_-2), uIdx_[(L-1)*L_] = L_*L_-1,
  dIdx_[(L-1)*L_] = L_*(L_-1)+1, pIdx_[(L-1)*L_] = 0, bIdx_[(L-1)*L_] = 1;
  // case i=L-1, j=L-1
  lIdx_[(L-1)*L_+L-1] = L_*(L_-2)+L_-2, rIdx_[(L-1)*L_+L-1] = L_*(L_-2)+L_-1, uIdx_[(L-1)*L_+L-1] = L_*L_-2,
  dIdx_[(L-1)*L_+L-1] = L_*(L_-1), pIdx_[(L-1)*L_+L-1] = L_-1, bIdx_[(L-1)*L_+L-1] = 0;
  // case i=0, j=1 ~ L-2
  for (int j=1; j<L_-1; ++j)
  {
    lIdx_[j] = L_*(L_-1)+j-1, rIdx_[j] = L_*(L_-1)+j, uIdx_[j] = j-1,
    dIdx_[j] = j+1, pIdx_[j] = L_+j, bIdx_[j] = L_+j+1;
  }
  // case i=L_-1, j=1 ~ L-2
  for (int j=1; j<L_-1; ++j)
  {
    lIdx_[(L_-1)*L_+j] = L_*(L_-2)+j-1, rIdx_[(L_-1)*L_+j] = L_*(L_-2)+j, uIdx_[(L_-1)*L_+j] = L_*(L_-1)+j-1,
    dIdx_[(L_-1)*L_+j] = L_*(L_-1)+j+1, pIdx_[(L_-1)*L_+j] = j, bIdx_[(L_-1)*L_+j] = j+1;
  }
  // case i= 1 ~ L-2, j=0
  for (int i=1; i<L_-1; ++i)
  {
    lIdx_[i*L_] = L_*(i-1)+L_-1, rIdx_[i*L_] = L_*(i-1), uIdx_[i*L_] = L_*i+L_-1,
    dIdx_[i*L_] = L_*i+1, pIdx_[i*L_] = L_*(i+1), bIdx_[i*L_] = L_*(i+1)+1;
  }
  // case i= 1 ~ L-2, j=L_-1
  for (int i=1; i<L_-1; ++i)
  {
    lIdx_[i*L_+L_-1] = L_*(i-1)+L_-2, rIdx_[i*L_+L_-1] = L_*(i-1)+L_-1, uIdx_[i*L_+L_-1] = L_*i+L_-2,
    dIdx_[i*L_+L_-1] = L_*i, pIdx_[i*L_+L_-1] = L_*(i+1)+L_-1, bIdx_[i*L_+L_-1] = L_*(i+1);
  }
  // Checkerboard link(To implement the MCMC update rule)
  for (int i=0; i<L*L; ++i)
    list_[i].set_item(i);
  // red board: (2*i+j)%3 == 0
  int idx0 = 0;
  for (int i=0; i<L_; ++i)
    for (int j=0; j<L_; ++j)
    {
      if ((2*i+j)%3 != 0)
        continue;
      const int idx1 = i*L_ + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // yellow board: (2*i+j)%3 == 1
  for (int i=0; i<L_; ++i)
    for (int j=0; j<L_; ++j)
    {
      if ((2*i+j)%3 != 1)
        continue;
      const int idx1 = i*L_ + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // green: (2*i+j)%3 == 2
  for (int i=0; i<L_; ++i)
    for (int j=0; j<L_; ++j)
    {
      if ((2*i+j)%3 != 2)
        continue;
      const int idx1 = i*L_ + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  list_[idx0].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename TraitsClass>
void TFITRI<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<nChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<L_; ++i)
      for (int j=0; j<L_; ++j)
      {
        const int idx = i*L_+j;
        diag_[k] += spinPtr[k*nSites+idx]*
                   (spinPtr[k*nSites+lIdx_[idx]]+spinPtr[k*nSites+rIdx_[idx]]+
                    spinPtr[k*nSites+uIdx_[idx]]+spinPtr[k*nSites+dIdx_[idx]]+
                    spinPtr[k*nSites+pIdx_[idx]]+spinPtr[k*nSites+bIdx_[idx]]);
      }
    diag_[k] *= 0.5;
  }
}

template <typename TraitsClass>
void TFITRI<TraitsClass>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename TraitsClass>
void TFITRI<TraitsClass>::accept_next_state(const std::vector<bool> & updateList)
{
  const int idx = idxptr_->get_item();
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  for (int k=0; k<nChains; ++k)
  {
    if (updateList[k])
      diag_[k] -= ktwo*spinPtr[k*nSites+idx]*
                   (spinPtr[k*nSites+lIdx_[idx]] +
                    spinPtr[k*nSites+rIdx_[idx]] +
                    spinPtr[k*nSites+uIdx_[idx]] +
                    spinPtr[k*nSites+dIdx_[idx]] +
                    spinPtr[k*nSites+pIdx_[idx]] +
                    spinPtr[k*nSites+bIdx_[idx]]);
  }
  machine_.spin_flip(updateList);
}

template <typename TraitsClass>
void TFITRI<TraitsClass>::get_htilda(std::complex<FloatType> * htilda)
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

template <typename TraitsClass>
void TFITRI<TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.backward(lnpsiGradients);
}

template <typename TraitsClass>
void TFITRI<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_variables(trueGradients, learningRate);
}
} // namespace spinhalfsystem
