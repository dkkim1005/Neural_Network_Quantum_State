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
      diag_[k] -= ktwo*spinPtr[k*nSites+idx].real()*(spinPtr[k*nSites+leftIdx_[idx]].real()+spinPtr[k*nSites+rightIdx_[idx]].real());
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
    htilda[k] = kJ*diag_[k].real();
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


template <typename TraitsClass>
TFICheckerBoard<TraitsClass>::TFICheckerBoard(AnsatzType & machine, const int L,
  const FloatType h, const std::array<FloatType, 2> Jarr,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<TFICheckerBoard, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
  kL(L),
  kh(h),
  kJ1(Jarr[0]),
  kJ2(Jarr[1]),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  Jmatrix_(machine.get_nInputs(), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
  nnidx_(machine.get_nInputs(), {0, 0, 0, 0, 0, 0, 0, 0})
{
  if (kL*kL != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L*L!");
  if (kL < 4 && kL%2 == 1)
    throw std::invalid_argument("The width of system is not adequate for constructing the checker board lattice.");
  /*
    index rule of nearest neighbors
       6  0  4
       2  x  3
       5  1  7
  */
  // open boundary condition
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      const int idx = i*kL+j;
      Jmatrix_[idx][0] = (i == 0) ? kzero : kJ1; // up
      Jmatrix_[idx][1] = (i == kL-1) ? kzero : kJ1; // down
      Jmatrix_[idx][2] = (j == 0) ? kzero : kJ1; // left
      Jmatrix_[idx][3] = (j == kL-1) ? kzero : kJ1; // right
      if ((i+j)%2 == 0)
      {
        Jmatrix_[idx][4] = (i == 0 || j == kL-1) ? kzero : kJ2; // up-right
        Jmatrix_[idx][5] = (i == kL-1 || j == 0) ? kzero : kJ2; // down-left
      }
      else
      {
        Jmatrix_[idx][6] = (i == 0 || j == 0) ? kzero : kJ2; // up-left
        Jmatrix_[idx][7] = (i == kL-1 || j == kL-1) ? kzero : kJ2; // down-right
      }
    }
  // table of the index for nearest-neighbor sites
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      const int idx = i*kL+j;
      nnidx_[idx][0] = (i == 0) ? (kL-1)*kL+j : (i-1)*kL+j; // up
      nnidx_[idx][1] = (i == kL-1) ? j : (i+1)*kL+j; // down
      nnidx_[idx][2] = (j == 0) ? i*kL+kL-1 : i*kL+j-1; // left
      nnidx_[idx][3] = (j == kL-1) ? i*kL : i*kL+j+1; // right
    }
  for (int i=1; i<kL-1; ++i)
    for (int j=1; j<kL-1; ++j)
    {
      const int idx = i*kL+j;
      nnidx_[idx][4] = (i-1)*kL+j+1; // up-right
      nnidx_[idx][5] = (i+1)*kL+j-1; // down-left
      nnidx_[idx][6] = (i-1)*kL+j-1; // up-left
      nnidx_[idx][7] = (i+1)*kL+j+1; // down-right
    }
  // i == 0, 1 <= j <= kL-2
  for (int j=1; j<kL-1; ++j)
  {
    const int idx = j;
    nnidx_[j][4] = (kL-1)*kL+j+1; // up-right
    nnidx_[j][5] = kL+j-1; // down-left
    nnidx_[j][6] = (kL-1)*kL+j-1; // up-left
    nnidx_[j][7] = kL+j+1; // down-right
  }
  // i == kL-1, 1 <= j <= kL-2
  for (int j=1; j<kL-1; ++j)
  {
    const int idx = (kL-1)*kL+j;
    nnidx_[idx][4] = (kL-2)*kL+j+1; // up-right
    nnidx_[idx][5] = j-1; // down-left
    nnidx_[idx][6] = (kL-2)*kL+j-1; // up-left
    nnidx_[idx][7] = j+1; // down-right
  }
  // 1 <= i <= kL-2, j == 0
  for (int i=1; i<kL-1; ++i)
  {
    const int idx = i*kL;
    nnidx_[idx][4] = (i-1)*kL+1; // up-right
    nnidx_[idx][5] = (i+1)*kL+kL-1; // down-left
    nnidx_[idx][6] = (i-1)*kL+kL-1; // up-left
    nnidx_[idx][7] = (i+1)*kL+1; // down-right
  }
  // 1 <= i <= kL-2, j == kL-1
  for (int i=1; i<kL-1; ++i)
  {
    const int idx = i*kL+kL-1;
    nnidx_[idx][4] = (i-1)*kL; // up-right
    nnidx_[idx][5] = (i+1)*kL+kL-2; // down-left
    nnidx_[idx][6] = (i-1)*kL+kL-2; // up-left
    nnidx_[idx][7] = (i+1)*kL; // down-right
  }
  // i == 0, j == 0
  nnidx_[0][4] = (kL-1)*kL+1; // up-right
  nnidx_[0][5] = kL+kL-1; // down-left
  nnidx_[0][6] = (kL-1)*kL+kL-1; // up-left
  nnidx_[0][7] = kL+1; // down-right
  // i == 0, j == kL-1
  nnidx_[kL-1][4] = (kL-1)*kL; // up-right
  nnidx_[kL-1][5] = kL+kL-2; // down-left
  nnidx_[kL-1][6] = (kL-1)*kL+kL-2; // up-left
  nnidx_[kL-1][7] = kL; // down-right
  // i == kL-1, j == 0
  nnidx_[(kL-1)*kL][4] = (kL-2)*kL+1; // up-right
  nnidx_[(kL-1)*kL][5] = kL-1; // down-left
  nnidx_[(kL-1)*kL][6] = (kL-2)*kL+kL-1; // up-left
  nnidx_[(kL-1)*kL][7] = 1; // down-right
  // i == kL-1, j == kL-1
  nnidx_[(kL-1)*kL+kL-1][4] = (kL-2)*kL; // up-right
  nnidx_[(kL-1)*kL+kL-1][5] = kL-2; // down-left
  nnidx_[(kL-1)*kL+kL-1][6] = (kL-2)*kL+kL-2; // up-left
  nnidx_[(kL-1)*kL+kL-1][7] = 0; // down-right
  // Checkerboard link(To implement the MCMC update rule)
  for (int i=0; i<kL*kL; ++i)
    list_[i].set_item(i);
  // white board: (i+j)%2 == 0
  int idx0 = 0;
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((i+j)%2 == 1)
        continue;
      const int idx1 = i*kL + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((i+j)%2 == 0)
        continue;
      const int idx1 = i*kL + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  list_[idx0].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<nChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<kL; ++i)
      for (int j=0; j<kL; ++j)
      {
        const int idx = i*kL+j;
        diag_[k] += spinPtr[k*nSites+idx]*
                   (spinPtr[k*nSites+nnidx_[idx][0]]*Jmatrix_[idx][0]+
                    spinPtr[k*nSites+nnidx_[idx][1]]*Jmatrix_[idx][1]+
                    spinPtr[k*nSites+nnidx_[idx][2]]*Jmatrix_[idx][2]+
                    spinPtr[k*nSites+nnidx_[idx][3]]*Jmatrix_[idx][3]+
                    spinPtr[k*nSites+nnidx_[idx][4]]*Jmatrix_[idx][4]+
                    spinPtr[k*nSites+nnidx_[idx][5]]*Jmatrix_[idx][5]+
                    spinPtr[k*nSites+nnidx_[idx][6]]*Jmatrix_[idx][6]+
                    spinPtr[k*nSites+nnidx_[idx][7]]*Jmatrix_[idx][7]);
      }
    diag_[k] *= 0.5;
  }
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::sampling(std::complex<FloatType> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::accept_next_state(const std::vector<bool> & updateList)
{
  const int idx = idxptr_->get_item();
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  for (int k=0; k<nChains; ++k)
  {
    if (updateList[k])
      diag_[k] -= ktwo*spinPtr[k*nSites+idx].real()*
                 (spinPtr[k*nSites+nnidx_[idx][0]].real()*Jmatrix_[idx][0]+
                  spinPtr[k*nSites+nnidx_[idx][1]].real()*Jmatrix_[idx][1]+
                  spinPtr[k*nSites+nnidx_[idx][2]].real()*Jmatrix_[idx][2]+
                  spinPtr[k*nSites+nnidx_[idx][3]].real()*Jmatrix_[idx][3]+
                  spinPtr[k*nSites+nnidx_[idx][4]].real()*Jmatrix_[idx][4]+
                  spinPtr[k*nSites+nnidx_[idx][5]].real()*Jmatrix_[idx][5]+
                  spinPtr[k*nSites+nnidx_[idx][6]].real()*Jmatrix_[idx][6]+
                  spinPtr[k*nSites+nnidx_[idx][7]].real()*Jmatrix_[idx][7]);
  }
  machine_.spin_flip(updateList);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::get_htilda(std::complex<FloatType> * htilda)
{
  /*
     htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
      --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}
   */
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  for (int k=0; k<nChains; ++k)
    htilda[k] = diag_[k].real();
  for (int i=0; i<nSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<nChains; ++k)
      htilda[k] += kh*std::exp(lnpsi1_[k] - lnpsi0_[k]);
  }
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.backward(lnpsiGradients);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_variables(trueGradients, learningRate);
}
} // namespace spinhalfsystem
