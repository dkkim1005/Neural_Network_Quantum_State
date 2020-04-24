// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

namespace spinhalf
{
template <typename TraitsClass>
TFISQ<TraitsClass>::TFISQ(AnsatzType & machine, const int L,
  const FloatType h, const FloatType J, const unsigned long long seedNumber, const FloatType dropOutRate):
  BaseParallelSampler<TFISQ, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedNumber),
  kL(L),
  knSites(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kgpuBlockSize(1+(machine.get_nChains()-1)/NUM_THREADS_PER_BLOCK),
  kh(h),
  kJ(J),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_dev_(machine.get_nChains()),
  nnidx_dev_(4*machine.get_nInputs()),
  batchAllocater_(machine.get_nHiddens(), dropOutRate)
{
  if (kL*kL != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L*L!");
  thrust::host_vector<int> nnidx_host(4*(kL*kL)); 
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      const int siteIdx = i*kL+j;
      nnidx_host[4*siteIdx+0] = (j!=0) ? kL*i+j-1 : kL*i+kL-1;
      nnidx_host[4*siteIdx+1] = (j!=kL-1) ? kL*i+j+1 : kL*i;
      nnidx_host[4*siteIdx+2] = (i!=0) ? kL*(i-1)+j : kL*(kL-1)+j;
      nnidx_host[4*siteIdx+3] = (i!=kL-1) ? kL*(i+1)+j : j;
    }
  nnidx_dev_ = nnidx_host;
  // Checkerboard link(To implement the MCMC update rule)
  for (int i=0; i<kL*kL; ++i)
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
void TFISQ<TraitsClass>::initialize(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.initialize(lnpsi_dev);
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  // diag_ = \sum_i spin_i*spin_{i+1}
  gpu_kernel::TFI__InitDiag__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL),
    PTR_FROM_THRUST(nnidx_dev_.data()), 4, PTR_FROM_THRUST(diag_dev_.data()));
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::sampling(thrust::complex<FloatType> * lnpsi_dev)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi_dev);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::accept_next_state(bool * isNewStateAccepted_dev)
{
  const int idx = idxptr_->get_item();
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  gpu_kernel::TFI__UpdateDiag__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL), PTR_FROM_THRUST(nnidx_dev_.data()), 4,
    isNewStateAccepted_dev, idx, PTR_FROM_THRUST(diag_dev_.data()));
  machine_.spin_flip(isNewStateAccepted_dev);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::get_htilda(const thrust::complex<FloatType> * lnpsi0_dev, thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev)
{
  /*
     htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
      --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}
   */
  gpu_kernel::common__SetValues__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(htilda_dev, knChains, kzero);
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, lnpsi1_dev);
    gpu_kernel::TFI__GetHtildaStep1__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, kh, lnpsi1_dev, lnpsi0_dev, htilda_dev);
  }
  gpu_kernel::TFI__GetHtildaStep2__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, kJ, PTR_FROM_THRUST(diag_dev_.data()), htilda_dev);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::get_lnpsiGradients(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  machine_.backward(lnpsiGradients_dev, batchAllocater_.get_miniBatch());
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::evolve(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate)
{
  machine_.update_variables(trueGradients_dev, learningRate, batchAllocater_.get_miniBatch());
  batchAllocater_.next();
}

/*
template <typename TraitsClass>
TFICheckerBoard<TraitsClass>::TFICheckerBoard(AnsatzType & machine, const int L,
  const FloatType h, const std::array<FloatType, 2> J1_J2, const bool isPeriodicBoundary,
  const unsigned long seedDistance, const unsigned long seedNumber, const FloatType dropOutRate):
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
  Jmatrix_(machine.get_nInputs(), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
  batchAllocater_(machine.get_nHiddens(), dropOutRate)
{
  if (kL*kL != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L*L!");
  if (kL < 4 && kL%2 == 1)
    throw std::invalid_argument("The width of system is not adequate for constructing the checker board lattice.");
  //  index rule of nearest neighbors
  //     6  0  4
  //     2  x  3
  //     5  1  7
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
  //   htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
  //    --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}
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
  machine_.partial_backward(lnpsiGradients, batchAllocater_.get_miniBatch());
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate)
{
  machine_.update_partial_variables(trueGradients, learningRate, batchAllocater_.get_miniBatch());
  batchAllocater_.next();
}
*/
} // namespace spinhalf

namespace gpu_kernel
{
template <typename FloatType>
__global__ void TFI__InitDiag__(
  const thrust::complex<FloatType> * spinStates,
  const int nChains,
  const int nSites,
  const int * nnidx,
  const int nnn,
  FloatType * diag)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType zero = 0.0;
  while (idx < nChains)
  {
    diag[idx] = zero;
    for (int i=0; i<nSites; ++i)
    {
      const int kstate = idx*nSites;
      FloatType sumNNSpins = zero;
      for (int n=0; n<nnn; ++n)
        sumNNSpins += spinStates[kstate+nnidx[nnn*i+n]].real();
      diag[idx] += spinStates[kstate+i].real()*sumNNSpins;
    }
    diag[idx] *= 0.5;
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void TFI__UpdateDiag__(
  const thrust::complex<FloatType> * spinStates,
  const int nChains,
  const int nSites,
  const int * nnidx,
  const int nnn,
  const bool * updateList,
  const int siteIdx,
  FloatType * diag)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType twoDelta[2] = {0.0, 2.0}, zero = 0.0;
  while (idx < nChains)
  {
    const int kstate = idx*nSites;
    FloatType sumNNSpins = zero;
    for (int n=0; n<nnn; ++n)
      sumNNSpins += spinStates[kstate+nnidx[nnn*siteIdx+n]].real();
    diag[idx] -= twoDelta[updateList[idx]]*spinStates[kstate+siteIdx].real()*sumNNSpins;
    idx += nstep;
  }
}

// htilda[k] += hfield*exp(lnpsi1[k] - lnpsi0[k]);
template <typename FloatType>
__global__ void TFI__GetHtildaStep1__(
  const int nChains,
  const FloatType hfield,
  const thrust::complex<FloatType> * lnpsi1,
  const thrust::complex<FloatType> * lnpsi0,
  thrust::complex<FloatType> * htilda)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    htilda[idx] += hfield*thrust::exp(lnpsi1[idx]-lnpsi0[idx]);
    idx += nstep;
  }
}

// htilda[k] += J*diag[k];
template <typename FloatType>
__global__ void TFI__GetHtildaStep2__(
  const int nChains,
  const FloatType J,
  const FloatType * diag,
  thrust::complex<FloatType> * htilda)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    htilda[idx] += J*diag[idx];
    idx += nstep;
  }
}
} // namespace gpu_kernel
