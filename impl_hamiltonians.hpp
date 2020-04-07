// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

namespace spinhalf
{
template <typename TraitsClass>
TFIChain<TraitsClass>::TFIChain(AnsatzType & machine, const FloatType h,
  const FloatType J, const unsigned long seedDistance, const unsigned long seedNumber, const FloatType dropOutRate):
  BaseParallelSampler<TFIChain, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
  knSites(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kh(h),
  kJ(J),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  nnidx_(machine.get_nInputs()),
  batchAllocater_(machine.get_nHiddens(), dropOutRate)
{
  // (checker board update) list_ : 1,3,5,...,2,4,6,...
  for (int i=0; i<knSites; i++)
    list_[i].set_item(i);
  int idx = 0;
  for (int i=2; i<knSites; i+=2)
  {
    list_[idx].set_nextptr(&list_[i]);
    idx = i;
  }
  for (int i=1; i<knSites; i+=2)
  {
    list_[idx].set_nextptr(&list_[i]);
    idx = i;
  }
  list_[idx].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
  // indexing of the nearest neighbor site(periodic boundary condition)
  for (int i=0; i<knSites; ++i)
  {
    nnidx_[i][0] = ((i==0) ? knSites-1 : i-1);
    nnidx_[i][1] = ((i==knSites-1) ? 0 : i+1);
  }
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<knChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<knSites-1; ++i)
      diag_[k] += spinPtr[k*knSites+i].real()*spinPtr[k*knSites+i+1].real();
    diag_[k] += spinPtr[k*knSites+knSites-1].real()*spinPtr[k*knSites].real();
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
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    if (updateList[k])
    {
      const int kknsites = k*knSites;
      diag_[k] -= ktwo*spinPtr[kknsites+idx].real()*(spinPtr[kknsites+nnidx_[idx][0]].real()+spinPtr[kknsites+nnidx_[idx][1]].real());
    }
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
  for (int k=0; k<knChains; ++k)
    htilda[k] = kJ*diag_[k];
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChains; ++k)
      htilda[k] += kh*std::exp(lnpsi1_[k] - lnpsi0_[k]);
  }
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.partial_backward(lnpsiGradients, batchAllocater_.get_miniBatch());
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_partial_variables(trueGradients, learningRate, batchAllocater_.get_miniBatch());
  batchAllocater_.next();
}


template <typename TraitsClass>
TFISQ<TraitsClass>::TFISQ(AnsatzType & machine, const int L,
  const FloatType h, const FloatType J, const unsigned long seedDistance,
  const unsigned long seedNumber, const FloatType dropOutRate):
  BaseParallelSampler<TFISQ, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
  kL(L),
  knSites(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kh(h),
  kJ(J),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  nnidx_(machine.get_nChains()),
  batchAllocater_(machine.get_nHiddens(), dropOutRate)
{
  if (kL*kL != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L*L!");
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      nnidx_[i*kL+j][0] = (j!=0) ? kL*i+j-1 : kL*i+kL-1;
      nnidx_[i*kL+j][1] = (j!=kL-1) ? kL*i+j+1 : kL*i;
      nnidx_[i*kL+j][2] = (i!=0) ? kL*(i-1)+j : kL*(kL-1)+j;
      nnidx_[i*kL+j][3] = (i!=kL-1) ? kL*(i+1)+j : j;
    }
  // Checkerboard link(To implement the MCMC update rule)
  for (int i=0; i<L*L; ++i)
    list_[i].set_item(i);
  // black board: (i+j)%2 == 0
  int idx0 = 0;
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((i+j)%2 != 0)
        continue;
      const int idx1 = i*kL+j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // white board: (i+j)%2 == 1
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((i+j)%2 != 1)
        continue;
      const int idx1 = i*kL+j;
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
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<knChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<kL; ++i)
      for (int j=0; j<kL; ++j)
      {
        const int idx = i*kL+j, kknsites = k*knSites;
        diag_[k] += spinPtr[kknsites+idx].real()*
                   (spinPtr[kknsites+nnidx_[idx][0]].real()+spinPtr[kknsites+nnidx_[idx][1]].real()+
                    spinPtr[kknsites+nnidx_[idx][2]].real()+spinPtr[kknsites+nnidx_[idx][3]].real());
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
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    if (updateList[k])
    {
      const int kknsites = k*knSites;
      diag_[k] -= ktwo*spinPtr[kknsites+idx].real()*
                   (spinPtr[kknsites+nnidx_[idx][0]].real() +
                    spinPtr[kknsites+nnidx_[idx][1]].real() +
                    spinPtr[kknsites+nnidx_[idx][2]].real() +
                    spinPtr[kknsites+nnidx_[idx][3]].real());
    }
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
  for (int k=0; k<knChains; ++k)
    htilda[k] = kJ*diag_[k];
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChains; ++k)
      htilda[k] += kh*std::exp(lnpsi1_[k] - lnpsi0_[k]);
  }
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.partial_backward(lnpsiGradients, batchAllocater_.get_miniBatch());
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_partial_variables(trueGradients, learningRate, batchAllocater_.get_miniBatch());
  batchAllocater_.next();
}


template <typename TraitsClass>
TFITRI<TraitsClass>::TFITRI(AnsatzType & machine, const int L,
  const FloatType h, const FloatType J,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<TFITRI, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
  kL(L),
  knSites(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kh(h),
  kJ(J),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  nnidx_(machine.get_nChains())
{
  if (kL*kL != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L*L!");
  for (int i=1; i<kL-1; ++i)
    for (int j=1; j<kL-1; ++j)
    {
      nnidx_[i*kL+j][0] = kL*(i-1)+j-1;
      nnidx_[i*kL+j][1] = kL*(i-1)+j;
      nnidx_[i*kL+j][2] = kL*i+j-1;
      nnidx_[i*kL+j][3] = kL*i+j+1;
      nnidx_[i*kL+j][4] = kL*(i+1)+j;
      nnidx_[i*kL+j][5] = kL*(i+1)+j+1;
    }
  // case i=0, j=0
  nnidx_[0][0] = kL*kL-1;
  nnidx_[0][1] = kL*(kL-1);
  nnidx_[0][2] = kL-1;
  nnidx_[0][3] = 1;
  nnidx_[0][4] = kL;
  nnidx_[0][5] = kL+1;
  // case i=0, j=L-1
  nnidx_[L-1][0] = kL*kL-2;
  nnidx_[L-1][1] = kL*kL-1;
  nnidx_[L-1][2] = kL-2;
  nnidx_[L-1][3] = 0;
  nnidx_[L-1][4] = kL+kL-1;
  nnidx_[L-1][5] = kL;
  // case i=L-1, j=0
  nnidx_[(L-1)*kL][0] = kL*(kL-2)+kL-1;
  nnidx_[(L-1)*kL][1] = kL*(kL-2);
  nnidx_[(L-1)*kL][2] = kL*kL-1;
  nnidx_[(L-1)*kL][3] = kL*(kL-1)+1;
  nnidx_[(L-1)*kL][4] = 0;
  nnidx_[(L-1)*kL][5] = 1;
  // case i=L-1, j=L-1
  nnidx_[(L-1)*kL+L-1][0] = kL*(kL-2)+kL-2;
  nnidx_[(L-1)*kL+L-1][1] = kL*(kL-2)+kL-1;
  nnidx_[(L-1)*kL+L-1][2] = kL*kL-2;
  nnidx_[(L-1)*kL+L-1][3] = kL*(kL-1);
  nnidx_[(L-1)*kL+L-1][4] = kL-1;
  nnidx_[(L-1)*kL+L-1][5] = 0;
  // case i=0, j=1 ~ L-2
  for (int j=1; j<kL-1; ++j)
  {
    nnidx_[j][0] = kL*(kL-1)+j-1;
    nnidx_[j][1] = kL*(kL-1)+j;
    nnidx_[j][2] = j-1;
    nnidx_[j][3] = j+1;
    nnidx_[j][4] = kL+j;
    nnidx_[j][5] = kL+j+1;
  }
  // case i=kL-1, j=1 ~ L-2
  for (int j=1; j<kL-1; ++j)
  {
    nnidx_[(kL-1)*kL+j][0] = kL*(kL-2)+j-1;
    nnidx_[(kL-1)*kL+j][1] = kL*(kL-2)+j;
    nnidx_[(kL-1)*kL+j][2] = kL*(kL-1)+j-1;
    nnidx_[(kL-1)*kL+j][3] = kL*(kL-1)+j+1;
    nnidx_[(kL-1)*kL+j][4] = j;
    nnidx_[(kL-1)*kL+j][5] = j+1;
  }
  // case i= 1 ~ L-2, j=0
  for (int i=1; i<kL-1; ++i)
  {
    nnidx_[i*kL][0] = kL*(i-1)+kL-1;
    nnidx_[i*kL][1] = kL*(i-1);
    nnidx_[i*kL][2] = kL*i+kL-1;
    nnidx_[i*kL][3] = kL*i+1;
    nnidx_[i*kL][4] = kL*(i+1);
    nnidx_[i*kL][5] = kL*(i+1)+1;
  }
  // case i= 1 ~ L-2, j=kL-1
  for (int i=1; i<kL-1; ++i)
  {
    nnidx_[i*kL+kL-1][0] = kL*(i-1)+kL-2;
    nnidx_[i*kL+kL-1][1] = kL*(i-1)+kL-1;
    nnidx_[i*kL+kL-1][2] = kL*i+kL-2;
    nnidx_[i*kL+kL-1][3] = kL*i;
    nnidx_[i*kL+kL-1][4] = kL*(i+1)+kL-1;
    nnidx_[i*kL+kL-1][5] = kL*(i+1);
  }
  // Checkerboard link(To implement the MCMC update rule)
  for (int i=0; i<L*L; ++i)
    list_[i].set_item(i);
  // red board: (2*i+j)%3 == 0
  int idx0 = 0;
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((2*i+j)%3 != 0)
        continue;
      const int idx1 = i*kL + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // yellow board: (2*i+j)%3 == 1
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((2*i+j)%3 != 1)
        continue;
      const int idx1 = i*kL + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // green: (2*i+j)%3 == 2
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((2*i+j)%3 != 2)
        continue;
      const int idx1 = i*kL + j;
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
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<knChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<kL; ++i)
      for (int j=0; j<kL; ++j)
      {
        const int idx = i*kL+j, kknsites = k*knSites;
        diag_[k] += spinPtr[kknsites+idx].real()*
                   (spinPtr[kknsites+nnidx_[idx][0]].real()+spinPtr[kknsites+nnidx_[idx][1]].real()+
                    spinPtr[kknsites+nnidx_[idx][2]].real()+spinPtr[kknsites+nnidx_[idx][3]].real()+
                    spinPtr[kknsites+nnidx_[idx][4]].real()+spinPtr[kknsites+nnidx_[idx][5]].real());
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
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    if (updateList[k])
    {
      const int kknsites = k*knSites;
      diag_[k] -= ktwo*spinPtr[kknsites+idx].real()*
                   (spinPtr[kknsites+nnidx_[idx][0]].real() +
                    spinPtr[kknsites+nnidx_[idx][1]].real() +
                    spinPtr[kknsites+nnidx_[idx][2]].real() +
                    spinPtr[kknsites+nnidx_[idx][3]].real() +
                    spinPtr[kknsites+nnidx_[idx][4]].real() +
                    spinPtr[kknsites+nnidx_[idx][5]].real());
    }
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
  for (int k=0; k<knChains; ++k)
    htilda[k] = kJ*diag_[k];
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChains; ++k)
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
  const FloatType h, const std::array<FloatType, 2> J1_J2, const bool isPeriodicBoundary,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelSampler<TFICheckerBoard, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
  kL(L),
  knSites(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kh(h),
  kJ1(J1_J2[0]),
  kJ2(J1_J2[1]),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  nnidx_(machine.get_nInputs(), {0, 0, 0, 0, 0, 0, 0, 0}),
  Jmatrix_(machine.get_nInputs(), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
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
      Jmatrix_[idx][0] = (i == 0) ? (kJ1*isPeriodicBoundary) : kJ1; // up
      Jmatrix_[idx][1] = (i == kL-1) ? (kJ1*isPeriodicBoundary) : kJ1; // down
      Jmatrix_[idx][2] = (j == 0) ? (kJ1*isPeriodicBoundary) : kJ1; // left
      Jmatrix_[idx][3] = (j == kL-1) ? (kJ1*isPeriodicBoundary) : kJ1; // right
      if ((i+j)%2 == 0)
      {
        Jmatrix_[idx][4] = (i == 0 || j == kL-1) ? (kJ2*isPeriodicBoundary) : kJ2; // up-right
        Jmatrix_[idx][5] = (i == kL-1 || j == 0) ? (kJ2*isPeriodicBoundary) : kJ2; // down-left
      }
      else
      {
        Jmatrix_[idx][6] = (i == 0 || j == 0) ? (kJ2*isPeriodicBoundary) : kJ2; // up-left
        Jmatrix_[idx][7] = (i == kL-1 || j == kL-1) ? (kJ2*isPeriodicBoundary) : kJ2; // down-right
      }
    }
  // table of the index for the nearest-neighbor sites
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
  // black board: (i+j)%2 == 1
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
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<knChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<kL; ++i)
      for (int j=0; j<kL; ++j)
      {
        const int idx = i*kL+j, kknsites = k*knSites;
        diag_[k] += spinPtr[kknsites+idx].real()*
                   (spinPtr[kknsites+nnidx_[idx][0]].real()*Jmatrix_[idx][0]+
                    spinPtr[kknsites+nnidx_[idx][1]].real()*Jmatrix_[idx][1]+
                    spinPtr[kknsites+nnidx_[idx][2]].real()*Jmatrix_[idx][2]+
                    spinPtr[kknsites+nnidx_[idx][3]].real()*Jmatrix_[idx][3]+
                    spinPtr[kknsites+nnidx_[idx][4]].real()*Jmatrix_[idx][4]+
                    spinPtr[kknsites+nnidx_[idx][5]].real()*Jmatrix_[idx][5]+
                    spinPtr[kknsites+nnidx_[idx][6]].real()*Jmatrix_[idx][6]+
                    spinPtr[kknsites+nnidx_[idx][7]].real()*Jmatrix_[idx][7]);
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
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    if (updateList[k])
    {
      const int kknsites = k*knSites;
      diag_[k] -= ktwo*spinPtr[kknsites+idx].real()*
                 (spinPtr[kknsites+nnidx_[idx][0]].real()*Jmatrix_[idx][0]+
                  spinPtr[kknsites+nnidx_[idx][1]].real()*Jmatrix_[idx][1]+
                  spinPtr[kknsites+nnidx_[idx][2]].real()*Jmatrix_[idx][2]+
                  spinPtr[kknsites+nnidx_[idx][3]].real()*Jmatrix_[idx][3]+
                  spinPtr[kknsites+nnidx_[idx][4]].real()*Jmatrix_[idx][4]+
                  spinPtr[kknsites+nnidx_[idx][5]].real()*Jmatrix_[idx][5]+
                  spinPtr[kknsites+nnidx_[idx][6]].real()*Jmatrix_[idx][6]+
                  spinPtr[kknsites+nnidx_[idx][7]].real()*Jmatrix_[idx][7]);
    }
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
  for (int k=0; k<knChains; ++k)
    htilda[k] = diag_[k];
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChains; ++k)
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
} // namespace spinhalf


namespace paralleltempering
{
namespace spinhalf
{
template <typename TraitsClass>
TFIChain<TraitsClass>::TFIChain(AnsatzType & machine, const int nChainsPerBeta, const int nBeta, const FloatType h,
  const FloatType J, const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelTemperingSampler<TFIChain, TraitsClass>(machine.get_nInputs(), nChainsPerBeta, nBeta, seedDistance, seedNumber),
  knTotChains(machine.get_nChains()),
  knSites(machine.get_nInputs()),
  knChainsPerBeta(nChainsPerBeta),
  knBeta(nBeta),
  kh(h),
  kJ(J),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  nnidx_(machine.get_nInputs())
{
  if (nChainsPerBeta*nBeta != machine.get_nChains())
    throw std::invalid_argument("nChainsPerBeta*nBeta != machine.get_nChains()");
  // (checker board update) list_ : 1,3,5,...,2,4,6,...
  for (int i=0; i<knSites; i++)
    list_[i].set_item(i);
  int idx = 0;
  for (int i=2; i<knSites; i+=2)
  {
    list_[idx].set_nextptr(&list_[i]);
    idx = i;
  }
  for (int i=1; i<knSites; i+=2)
  {
    list_[idx].set_nextptr(&list_[i]);
    idx = i;
  }
  list_[idx].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
  // indexing of the nearest neighbor site(periodic boundary condition)
  for (int i=0; i<knSites; ++i)
  {
    nnidx_[i][0] = ((i==0) ? knSites-1 : i-1); // left
    nnidx_[i][1] = ((i==knSites-1) ? 0 : i+1); // right
  }
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::initialize(std::complex<FloatType> * lnpsi)
{
  machine_.initialize(lnpsi);
  const std::complex<FloatType> * spinPtr = machine_.get_spinStates();
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<knTotChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<knSites-1; ++i)
      diag_[k] += spinPtr[k*knSites+i].real()*spinPtr[k*knSites+i+1].real();
    diag_[k] += spinPtr[k*knSites+knSites-1].real()*spinPtr[k*knSites].real();
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
  #pragma omp parallel for
  for (int k=0; k<knTotChains; ++k)
  {
    if (updateList[k])
    {
      const int kknsites = k*knSites;
      diag_[k] -= ktwo*spinPtr[kknsites+idx].real()*(spinPtr[kknsites+nnidx_[idx][0]].real()+spinPtr[kknsites+nnidx_[idx][1]].real());
    }
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
  for (int k=0; k<knChainsPerBeta; ++k)
    htilda[k] = kJ*diag_[k];
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChainsPerBeta; ++k)
      htilda[k] += kh*std::exp(lnpsi1_[k] - lnpsi0_[k]);
  }
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.backward(lnpsiGradients, knChainsPerBeta);
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_variables(trueGradients, learningRate);
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::swap_states(const int & k1, const int & k2)
{
  machine_.swap_states(k1, k2);
  std::swap(diag_[k1], diag_[k2]);
}


template <typename TraitsClass>
TFITRI<TraitsClass>::TFITRI(AnsatzType & machine, const int L, const int nChainsPerBeta, const int nBeta,
  const FloatType h, const FloatType J, const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelTemperingSampler<TFITRI, TraitsClass>(machine.get_nInputs(), nChainsPerBeta, nBeta, seedDistance, seedNumber),
  kL(L),
  knSites(machine.get_nInputs()),
  knTotChains(nChainsPerBeta*nBeta),
  knChainsPerBeta(nChainsPerBeta),
  knBeta(nBeta),
  kh(h),
  kJ(J),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  nnidx_(machine.get_nChains())
{
  if (kL*kL != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L*L!");
  for (int i=1; i<kL-1; ++i)
    for (int j=1; j<kL-1; ++j)
    {
      nnidx_[i*kL+j][0] = kL*(i-1)+j-1;
      nnidx_[i*kL+j][1] = kL*(i-1)+j;
      nnidx_[i*kL+j][2] = kL*i+j-1;
      nnidx_[i*kL+j][3] = kL*i+j+1;
      nnidx_[i*kL+j][4] = kL*(i+1)+j;
      nnidx_[i*kL+j][5] = kL*(i+1)+j+1;
    }
  // case i=0, j=0
  nnidx_[0][0] = kL*kL-1;
  nnidx_[0][1] = kL*(kL-1);
  nnidx_[0][2] = kL-1;
  nnidx_[0][3] = 1;
  nnidx_[0][4] = kL;
  nnidx_[0][5] = kL+1;
  // case i=0, j=L-1
  nnidx_[L-1][0] = kL*kL-2;
  nnidx_[L-1][1] = kL*kL-1;
  nnidx_[L-1][2] = kL-2;
  nnidx_[L-1][3] = 0;
  nnidx_[L-1][4] = kL+kL-1;
  nnidx_[L-1][5] = kL;
  // case i=L-1, j=0
  nnidx_[(L-1)*kL][0] = kL*(kL-2)+kL-1;
  nnidx_[(L-1)*kL][1] = kL*(kL-2);
  nnidx_[(L-1)*kL][2] = kL*kL-1;
  nnidx_[(L-1)*kL][3] = kL*(kL-1)+1;
  nnidx_[(L-1)*kL][4] = 0;
  nnidx_[(L-1)*kL][5] = 1;
  // case i=L-1, j=L-1
  nnidx_[(L-1)*kL+L-1][0] = kL*(kL-2)+kL-2;
  nnidx_[(L-1)*kL+L-1][1] = kL*(kL-2)+kL-1;
  nnidx_[(L-1)*kL+L-1][2] = kL*kL-2;
  nnidx_[(L-1)*kL+L-1][3] = kL*(kL-1);
  nnidx_[(L-1)*kL+L-1][4] = kL-1;
  nnidx_[(L-1)*kL+L-1][5] = 0;
  // case i=0, j=1 ~ L-2
  for (int j=1; j<kL-1; ++j)
  {
    nnidx_[j][0] = kL*(kL-1)+j-1;
    nnidx_[j][1] = kL*(kL-1)+j;
    nnidx_[j][2] = j-1;
    nnidx_[j][3] = j+1;
    nnidx_[j][4] = kL+j;
    nnidx_[j][5] = kL+j+1;
  }
  // case i=kL-1, j=1 ~ L-2
  for (int j=1; j<kL-1; ++j)
  {
    nnidx_[(kL-1)*kL+j][0] = kL*(kL-2)+j-1;
    nnidx_[(kL-1)*kL+j][1] = kL*(kL-2)+j;
    nnidx_[(kL-1)*kL+j][2] = kL*(kL-1)+j-1;
    nnidx_[(kL-1)*kL+j][3] = kL*(kL-1)+j+1;
    nnidx_[(kL-1)*kL+j][4] = j;
    nnidx_[(kL-1)*kL+j][5] = j+1;
  }
  // case i= 1 ~ L-2, j=0
  for (int i=1; i<kL-1; ++i)
  {
    nnidx_[i*kL][0] = kL*(i-1)+kL-1;
    nnidx_[i*kL][1] = kL*(i-1);
    nnidx_[i*kL][2] = kL*i+kL-1;
    nnidx_[i*kL][3] = kL*i+1;
    nnidx_[i*kL][4] = kL*(i+1);
    nnidx_[i*kL][5] = kL*(i+1)+1;
  }
  // case i= 1 ~ L-2, j=kL-1
  for (int i=1; i<kL-1; ++i)
  {
    nnidx_[i*kL+kL-1][0] = kL*(i-1)+kL-2;
    nnidx_[i*kL+kL-1][1] = kL*(i-1)+kL-1;
    nnidx_[i*kL+kL-1][2] = kL*i+kL-2;
    nnidx_[i*kL+kL-1][3] = kL*i;
    nnidx_[i*kL+kL-1][4] = kL*(i+1)+kL-1;
    nnidx_[i*kL+kL-1][5] = kL*(i+1);
  }
  // Checkerboard link(To implement the MCMC update rule)
  for (int i=0; i<L*L; ++i)
    list_[i].set_item(i);
  // red board: (2*i+j)%3 == 0
  int idx0 = 0;
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((2*i+j)%3 != 0)
        continue;
      const int idx1 = i*kL + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // yellow board: (2*i+j)%3 == 1
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((2*i+j)%3 != 1)
        continue;
      const int idx1 = i*kL + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // green: (2*i+j)%3 == 2
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      if ((2*i+j)%3 != 2)
        continue;
      const int idx1 = i*kL + j;
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
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<knTotChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<kL; ++i)
      for (int j=0; j<kL; ++j)
      {
        const int idx = i*kL+j, kknsites = k*knSites;
        diag_[k] += spinPtr[kknsites+idx].real()*
                   (spinPtr[kknsites+nnidx_[idx][0]].real()+spinPtr[kknsites+nnidx_[idx][1]].real()+
                    spinPtr[kknsites+nnidx_[idx][2]].real()+spinPtr[kknsites+nnidx_[idx][3]].real()+
                    spinPtr[kknsites+nnidx_[idx][4]].real()+spinPtr[kknsites+nnidx_[idx][5]].real());
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
  #pragma omp parallel for
  for (int k=0; k<knTotChains; ++k)
  {
    if (updateList[k])
    {
      const int kknsites = k*knSites;
      diag_[k] -= ktwo*spinPtr[kknsites+idx].real()*
                   (spinPtr[kknsites+nnidx_[idx][0]].real() +
                    spinPtr[kknsites+nnidx_[idx][1]].real() +
                    spinPtr[kknsites+nnidx_[idx][2]].real() +
                    spinPtr[kknsites+nnidx_[idx][3]].real() +
                    spinPtr[kknsites+nnidx_[idx][4]].real() +
                    spinPtr[kknsites+nnidx_[idx][5]].real());
    }
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
  for (int k=0; k<knChainsPerBeta; ++k)
    htilda[k] = kJ*diag_[k];
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChainsPerBeta; ++k)
      htilda[k] += kh*std::exp(lnpsi1_[k] - lnpsi0_[k]);
  }
}

template <typename TraitsClass>
void TFITRI<TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.partial_backward(lnpsiGradients, knChainsPerBeta);
}

template <typename TraitsClass>
void TFITRI<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_variables(trueGradients, learningRate);
}

template <typename TraitsClass>
void TFITRI<TraitsClass>::swap_states(const int & k1, const int & k2)
{
  machine_.swap_states(k1, k2);
  std::swap(diag_[k1], diag_[k2]);
}


template <typename TraitsClass>
TFICheckerBoard<TraitsClass>::TFICheckerBoard(AnsatzType & machine, const int L,
  const int nChainsPerBeta, const int nBeta, const FloatType h,
  const std::array<FloatType, 2> J1_J2, const bool isPeriodicBoundary,
  const unsigned long seedDistance, const unsigned long seedNumber):
  BaseParallelTemperingSampler<TFICheckerBoard, TraitsClass>(machine.get_nInputs(), nChainsPerBeta, nBeta, seedDistance, seedNumber),
  kL(L),
  knSites(machine.get_nInputs()),
  knTotChains(nChainsPerBeta*nBeta),
  knChainsPerBeta(nChainsPerBeta),
  knBeta(nBeta),
  kh(h),
  kJ1(J1_J2[0]),
  kJ2(J1_J2[1]),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_(machine.get_nChains()),
  nnidx_(machine.get_nInputs(), {0, 0, 0, 0, 0, 0, 0, 0}),
  Jmatrix_(machine.get_nInputs(), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
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
      Jmatrix_[idx][0] = (i == 0) ? (kJ1*isPeriodicBoundary) : kJ1; // up
      Jmatrix_[idx][1] = (i == kL-1) ? (kJ1*isPeriodicBoundary) : kJ1; // down
      Jmatrix_[idx][2] = (j == 0) ? (kJ1*isPeriodicBoundary) : kJ1; // left
      Jmatrix_[idx][3] = (j == kL-1) ? (kJ1*isPeriodicBoundary) : kJ1; // right
      if ((i+j)%2 == 0)
      {
        Jmatrix_[idx][4] = (i == 0 || j == kL-1) ? (kJ2*isPeriodicBoundary) : kJ2; // up-right
        Jmatrix_[idx][5] = (i == kL-1 || j == 0) ? (kJ2*isPeriodicBoundary) : kJ2; // down-left
      }
      else
      {
        Jmatrix_[idx][6] = (i == 0 || j == 0) ? (kJ2*isPeriodicBoundary) : kJ2; // up-left
        Jmatrix_[idx][7] = (i == kL-1 || j == kL-1) ? (kJ2*isPeriodicBoundary) : kJ2; // down-right
      }
    }
  // table of the index for the nearest-neighbor sites
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
  // black board: (i+j)%2 == 1
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
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<knTotChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<kL; ++i)
      for (int j=0; j<kL; ++j)
      {
        const int idx = i*kL+j, kknsites = k*knSites;
        diag_[k] += spinPtr[kknsites+idx].real()*
                   (spinPtr[kknsites+nnidx_[idx][0]].real()*Jmatrix_[idx][0]+
                    spinPtr[kknsites+nnidx_[idx][1]].real()*Jmatrix_[idx][1]+
                    spinPtr[kknsites+nnidx_[idx][2]].real()*Jmatrix_[idx][2]+
                    spinPtr[kknsites+nnidx_[idx][3]].real()*Jmatrix_[idx][3]+
                    spinPtr[kknsites+nnidx_[idx][4]].real()*Jmatrix_[idx][4]+
                    spinPtr[kknsites+nnidx_[idx][5]].real()*Jmatrix_[idx][5]+
                    spinPtr[kknsites+nnidx_[idx][6]].real()*Jmatrix_[idx][6]+
                    spinPtr[kknsites+nnidx_[idx][7]].real()*Jmatrix_[idx][7]);
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
  #pragma omp parallel for
  for (int k=0; k<knTotChains; ++k)
  {
    if (updateList[k])
    {
      const int kknsites = k*knSites;
      diag_[k] -= ktwo*spinPtr[kknsites+idx].real()*
                 (spinPtr[kknsites+nnidx_[idx][0]].real()*Jmatrix_[idx][0]+
                  spinPtr[kknsites+nnidx_[idx][1]].real()*Jmatrix_[idx][1]+
                  spinPtr[kknsites+nnidx_[idx][2]].real()*Jmatrix_[idx][2]+
                  spinPtr[kknsites+nnidx_[idx][3]].real()*Jmatrix_[idx][3]+
                  spinPtr[kknsites+nnidx_[idx][4]].real()*Jmatrix_[idx][4]+
                  spinPtr[kknsites+nnidx_[idx][5]].real()*Jmatrix_[idx][5]+
                  spinPtr[kknsites+nnidx_[idx][6]].real()*Jmatrix_[idx][6]+
                  spinPtr[kknsites+nnidx_[idx][7]].real()*Jmatrix_[idx][7]);
    }
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
  for (int k=0; k<knChainsPerBeta; ++k)
    htilda[k] = diag_[k];
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChainsPerBeta; ++k)
      htilda[k] += kh*std::exp(lnpsi1_[k] - lnpsi0_[k]);
  }
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients)
{
  machine_.partial_backward(lnpsiGradients, knChainsPerBeta);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_variables(trueGradients, learningRate);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::swap_states(const int & k1, const int & k2)
{
  machine_.swap_states(k1, k2);
  std::swap(diag_[k1], diag_[k2]);
}
} // namespace spinhalf
} // namespace paralleltempering
