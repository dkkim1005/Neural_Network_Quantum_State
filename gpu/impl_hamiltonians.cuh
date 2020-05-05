// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

namespace spinhalf
{
template <typename TraitsClass>
TFISQ<TraitsClass>::TFISQ(AnsatzType & machine, const uint32_t L, const FloatType h, const FloatType J,
  const unsigned long seedNumber, const unsigned long seedDistance, const FloatType dropOutRate, const std::string prefix):
  BaseParallelSampler<TFISQ, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedNumber, seedDistance),
  kL(L),
  knSites(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kgpuBlockSize(1+(machine.get_nChains()-1)/NUM_THREADS_PER_BLOCK),
  kh(h),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_dev_(machine.get_nChains()),
  nnidx_dev_(4*machine.get_nInputs()),
  kJmatrix_dev(4*machine.get_nInputs(), J),
  kprefix(prefix),
  batchAllocater_(machine.get_nHiddens(), dropOutRate)
{
  if (kL*kL != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L*L!");
  thrust::host_vector<uint32_t> nnidx_host(4*(kL*kL)); 
  for (uint32_t i=0u; i<kL; ++i)
    for (uint32_t j=0u; j<kL; ++j)
    {
      const uint32_t siteIdx = i*kL+j;
      nnidx_host[4u*siteIdx+0u] = (j!=0u) ? kL*i+j-1u : kL*i+kL-1u;
      nnidx_host[4u*siteIdx+1u] = (j!=kL-1u) ? kL*i+j+1u : kL*i;
      nnidx_host[4u*siteIdx+2u] = (i!=0u) ? kL*(i-1u)+j : kL*(kL-1u)+j;
      nnidx_host[4u*siteIdx+3u] = (i!=kL-1u) ? kL*(i+1u)+j : j;
    }
  nnidx_dev_ = nnidx_host;
  // Checkerboard link(To implement the MCMC update rule)
  for (uint32_t i=0u; i<kL*kL; ++i)
    list_[i].set_item(i);
  // black board: (i+j)%2 == 0
  uint32_t idx0 = 0u;
  for (uint32_t i=0u; i<kL; ++i)
    for (uint32_t j=0u; j<kL; ++j)
    {
      if ((i+j)%2u != 0u)
        continue;
      const uint32_t idx1 = i*kL+j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // white board: (i+j)%2 == 1
  for (uint32_t i=0u; i<kL; ++i)
    for (uint32_t j=0u; j<kL; ++j)
    {
      if ((i+j)%2u != 1u)
        continue;
      const uint32_t idx1 = i*kL+j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  list_[idx0].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.initialize(lnpsi_dev);
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  // diag_ = \sum_i spin_i*spin_{i+1}
  gpu_kernel::TFI__GetDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL),
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(kJmatrix_dev.data()), 4u, PTR_FROM_THRUST(diag_dev_.data()));
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::sampling_(thrust::complex<FloatType> * lnpsi_dev)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi_dev);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::accept_next_state_(bool * isNewStateAccepted_dev)
{
  const uint32_t idx = idxptr_->get_item();
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  gpu_kernel::TFI__UpdateDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL),
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(kJmatrix_dev.data()), 4u, isNewStateAccepted_dev,
    idx, PTR_FROM_THRUST(diag_dev_.data()));
  machine_.spin_flip(isNewStateAccepted_dev);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::get_htilda_(const thrust::complex<FloatType> * lnpsi0_dev, thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev)
{
  /*
     htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
      --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}
   */
  gpu_kernel::common__copyFromRealToImag__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(diag_dev_.data()), knChains, htilda_dev);
  for (uint32_t i=0u; i<knSites; ++i)
  {
    machine_.forward(i, lnpsi1_dev);
    gpu_kernel::TFI__GetOffDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, kh, lnpsi1_dev, lnpsi0_dev, htilda_dev);
  }
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  machine_.backward(lnpsiGradients_dev, batchAllocater_.get_miniBatch());
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate)
{
  machine_.update_variables(trueGradients_dev, learningRate, batchAllocater_.get_miniBatch());
  batchAllocater_.next();
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::save_() const
{
  machine_.save(FNNDataType::W1, kprefix + "Dw1.dat");
  machine_.save(FNNDataType::W2, kprefix + "Dw2.dat");
  machine_.save(FNNDataType::B1, kprefix + "Db1.dat");
}


template <typename TraitsClass>
TFICheckerBoard<TraitsClass>::TFICheckerBoard(AnsatzType & machine, const uint32_t L, const FloatType h,
  const std::array<FloatType, 2> J1_J2, const bool isPeriodicBoundary, const unsigned long seedNumber,
  const unsigned long seedDistance, const FloatType dropOutRate, const std::string prefix):
  BaseParallelSampler<TFICheckerBoard, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedNumber, seedDistance),
  kL(L),
  knSites(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kgpuBlockSize(1+(machine.get_nChains()-1)/NUM_THREADS_PER_BLOCK),
  kh(h),
  kJ1(J1_J2[0]),
  kJ2(J1_J2[1]),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_dev_(machine.get_nChains()),
  nnidx_dev_(8*machine.get_nInputs(), 0),
  Jmatrix_dev_(8*machine.get_nInputs(), 0),
  kprefix(prefix),
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
  thrust::host_vector<FloatType> Jmatrix_host(8*kL*kL);
  thrust::host_vector<uint32_t> nnidx_host(8*kL*kL);
  for (uint32_t i=0u; i<kL; ++i)
    for (uint32_t j=0u; j<kL; ++j)
    {
      const uint32_t idx = i*kL+j;
      Jmatrix_host[8u*idx+0u] = (i == 0u) ? (kJ1*isPeriodicBoundary) : kJ1; // up
      Jmatrix_host[8u*idx+1u] = (i == kL-1u) ? (kJ1*isPeriodicBoundary) : kJ1; // down
      Jmatrix_host[8u*idx+2u] = (j == 0u) ? (kJ1*isPeriodicBoundary) : kJ1; // left
      Jmatrix_host[8u*idx+3u] = (j == kL-1u) ? (kJ1*isPeriodicBoundary) : kJ1; // right
      if ((i+j)%2u == 0u)
      {
        Jmatrix_host[8u*idx+4u] = (i == 0u || j == kL-1u) ? (kJ2*isPeriodicBoundary) : kJ2; // up-right
        Jmatrix_host[8u*idx+5u] = (i == kL-1u || j == 0u) ? (kJ2*isPeriodicBoundary) : kJ2; // down-left
      }
      else
      {
        Jmatrix_host[8u*idx+6u] = (i == 0u || j == 0u) ? (kJ2*isPeriodicBoundary) : kJ2; // up-left
        Jmatrix_host[8u*idx+7u] = (i == kL-1u || j == kL-1u) ? (kJ2*isPeriodicBoundary) : kJ2; // down-right
      }
    }
  Jmatrix_dev_ = Jmatrix_host;
  // table of the index for the nearest-neighbor sites
  for (uint32_t i=0u; i<kL; ++i)
    for (uint32_t j=0u; j<kL; ++j)
    {
      const uint32_t idx = i*kL+j;
      nnidx_host[8u*idx+0u] = (i == 0u) ? (kL-1u)*kL+j : (i-1u)*kL+j; // up
      nnidx_host[8u*idx+1u] = (i == kL-1u) ? j : (i+1u)*kL+j; // down
      nnidx_host[8u*idx+2u] = (j == 0u) ? i*kL+kL-1u : i*kL+j-1u; // left
      nnidx_host[8u*idx+3u] = (j == kL-1u) ? i*kL : i*kL+j+1u; // right
    }
  for (uint32_t i=1u; i<kL-1u; ++i)
    for (uint32_t j=1u; j<kL-1u; ++j)
    {
      const uint32_t idx = i*kL+j;
      nnidx_host[8u*idx+4u] = (i-1u)*kL+j+1u; // up-right
      nnidx_host[8u*idx+5u] = (i+1u)*kL+j-1u; // down-left
      nnidx_host[8u*idx+6u] = (i-1u)*kL+j-1u; // up-left
      nnidx_host[8u*idx+7u] = (i+1u)*kL+j+1u; // down-right
    }
  // i == 0, 1 <= j <= kL-2
  for (uint32_t j=1u; j<kL-1u; ++j)
  {
    const uint32_t idx = j;
    nnidx_host[8u*idx+4u] = (kL-1u)*kL+j+1u; // up-right
    nnidx_host[8u*idx+5u] = kL+j-1u; // down-left
    nnidx_host[8u*idx+6u] = (kL-1u)*kL+j-1u; // up-left
    nnidx_host[8u*idx+7u] = kL+j+1u; // down-right
  }
  // i == kL-1, 1 <= j <= kL-2
  for (uint32_t j=1u; j<kL-1u; ++j)
  {
    const uint32_t idx = (kL-1u)*kL+j;
    nnidx_host[8u*idx+4u] = (kL-2u)*kL+j+1u; // up-right
    nnidx_host[8u*idx+5u] = j-1u; // down-left
    nnidx_host[8u*idx+6u] = (kL-2u)*kL+j-1u; // up-left
    nnidx_host[8u*idx+7u] = j+1u; // down-right
  }
  // 1 <= i <= kL-2, j == 0
  for (uint32_t i=1u; i<kL-1u; ++i)
  {
    const uint32_t idx = i*kL;
    nnidx_host[8u*idx+4u] = (i-1u)*kL+1u; // up-right
    nnidx_host[8u*idx+5u] = (i+1u)*kL+kL-1u; // down-left
    nnidx_host[8u*idx+6u] = (i-1u)*kL+kL-1u; // up-left
    nnidx_host[8u*idx+7u] = (i+1u)*kL+1u; // down-right
  }
  // 1 <= i <= kL-2, j == kL-1
  for (uint32_t i=1u; i<kL-1u; ++i)
  {
    const uint32_t idx = i*kL+kL-1u;
    nnidx_host[8u*idx+4u] = (i-1u)*kL; // up-right
    nnidx_host[8u*idx+5u] = (i+1u)*kL+kL-2u; // down-left
    nnidx_host[8u*idx+6u] = (i-1u)*kL+kL-2u; // up-left
    nnidx_host[8u*idx+7u] = (i+1u)*kL; // down-right
  }
  // i == 0, j == 0
  nnidx_host[4] = (kL-1u)*kL+1u; // up-right
  nnidx_host[5] = kL+kL-1u; // down-left
  nnidx_host[6] = (kL-1u)*kL+kL-1u; // up-left
  nnidx_host[7] = kL+1u; // down-right
  // i == 0, j == kL-1
  nnidx_host[8u*(kL-1u)+4u] = (kL-1u)*kL; // up-right
  nnidx_host[8u*(kL-1u)+5u] = kL+kL-2u; // down-left
  nnidx_host[8u*(kL-1u)+6u] = (kL-1u)*kL+kL-2u; // up-left
  nnidx_host[8u*(kL-1u)+7u] = kL; // down-right
  // i == kL-1, j == 0
  nnidx_host[8u*(kL-1u)*kL+4u] = (kL-2u)*kL+1u; // up-right
  nnidx_host[8u*(kL-1u)*kL+5u] = kL-1u; // down-left
  nnidx_host[8u*(kL-1u)*kL+6u] = (kL-2u)*kL+kL-1u; // up-left
  nnidx_host[8u*(kL-1u)*kL+7u] = 1u; // down-right
  // i == kL-1, j == kL-1
  nnidx_host[8u*((kL-1u)*kL+kL-1u)+4u] = (kL-2u)*kL; // up-right
  nnidx_host[8u*((kL-1u)*kL+kL-1u)+5u] = kL-2u; // down-left
  nnidx_host[8u*((kL-1u)*kL+kL-1u)+6u] = (kL-2u)*kL+kL-2u; // up-left
  nnidx_host[8u*((kL-1u)*kL+kL-1u)+7u] = 0u; // down-right
  nnidx_dev_ = nnidx_host;
  // Checkerboard link(To implement the MCMC update rule)
  for (uint32_t i=0u; i<kL*kL; ++i)
    list_[i].set_item(i);
  // white board: (i+j)%2 == 0
  uint32_t idx0 = 0u;
  for (uint32_t i=0u; i<kL; ++i)
    for (uint32_t j=0u; j<kL; ++j)
    {
      if ((i+j)%2u == 1u)
        continue;
      const uint32_t idx1 = i*kL + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  // black board: (i+j)%2 == 1
  for (uint32_t i=0u; i<kL; ++i)
    for (uint32_t j=0u; j<kL; ++j)
    {
      if ((i+j)%2u == 0u)
        continue;
      const uint32_t idx1 = i*kL + j;
      list_[idx0].set_nextptr(&list_[idx1]);
      idx0 = idx1;
    }
  list_[idx0].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.initialize(lnpsi_dev);
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  // diag_ = \sum_i spin_i*spin_{i+1}
  gpu_kernel::TFI__GetDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL),
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(Jmatrix_dev_.data()), 8u, PTR_FROM_THRUST(diag_dev_.data()));
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::sampling_(thrust::complex<FloatType> * lnpsi_dev)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi_dev);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::accept_next_state_(bool * isNewStateAccepted_dev)
{
  const uint32_t idx = idxptr_->get_item();
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  gpu_kernel::TFI__UpdateDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL),
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(Jmatrix_dev_.data()), 8u, isNewStateAccepted_dev,
    idx, PTR_FROM_THRUST(diag_dev_.data()));
  machine_.spin_flip(isNewStateAccepted_dev);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::get_htilda_(const thrust::complex<FloatType> * lnpsi0_dev, thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev)
{
  //   htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
  //    --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}
  gpu_kernel::common__copyFromRealToImag__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(diag_dev_.data()), knChains, htilda_dev);
  for (uint32_t i=0u; i<knSites; ++i)
  {
    machine_.forward(i, lnpsi1_dev);
    gpu_kernel::TFI__GetOffDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, kh, lnpsi1_dev, lnpsi0_dev, htilda_dev);
  }
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  machine_.backward(lnpsiGradients_dev, batchAllocater_.get_miniBatch());
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate)
{
  machine_.update_variables(trueGradients_dev, learningRate, batchAllocater_.get_miniBatch());
  batchAllocater_.next();
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::save_() const
{
  machine_.save(FNNDataType::W1, kprefix + "Dw1.dat");
  machine_.save(FNNDataType::W2, kprefix + "Dw2.dat");
  machine_.save(FNNDataType::B1, kprefix + "Db1.dat");
}
} // namespace spinhalf

namespace gpu_kernel
{
template <typename FloatType>
__global__ void TFI__GetDiagElem__(
  const thrust::complex<FloatType> * spinStates,
  const uint32_t nChains,
  const uint32_t nSites,
  const uint32_t * nnidx,
  const FloatType * Jmatrix,
  const uint32_t nnn,
  FloatType * diag)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType zero = 0.0;
  while (idx < nChains)
  {
    diag[idx] = zero;
    for (uint32_t i=0u; i<nSites; ++i)
    {
      const uint32_t kstate = idx*nSites;
      FloatType sumNNInteractions = zero;
      for (uint32_t n=0u; n<nnn; ++n)
        sumNNInteractions += spinStates[kstate+nnidx[nnn*i+n]].real()*Jmatrix[nnn*i+n];
      diag[idx] += spinStates[kstate+i].real()*sumNNInteractions;
    }
    diag[idx] *= 0.5;
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void TFI__UpdateDiagElem__(
  const thrust::complex<FloatType> * spinStates,
  const uint32_t nChains,
  const uint32_t nSites,
  const uint32_t * nnidx,
  const FloatType * Jmatrix,
  const uint32_t nnn,
  const bool * updateList,
  const uint32_t siteIdx,
  FloatType * diag)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType twoDelta[2] = {0.0, 2.0}, zero = 0.0;
  while (idx < nChains)
  {
    const uint32_t kstate = idx*nSites;
    FloatType sumNNInteractions = zero;
    for (uint32_t n=0u; n<nnn; ++n)
      sumNNInteractions += spinStates[kstate+nnidx[nnn*siteIdx+n]].real()*Jmatrix[nnn*siteIdx+n];
    diag[idx] -= twoDelta[updateList[idx]]*spinStates[kstate+siteIdx].real()*sumNNInteractions;
    idx += nstep;
  }
}

// htilda[k] += hfield*exp(lnpsi1[k] - lnpsi0[k]);
template <typename FloatType>
__global__ void TFI__GetOffDiagElem__(
  const uint32_t nChains,
  const FloatType hfield,
  const thrust::complex<FloatType> * lnpsi1,
  const thrust::complex<FloatType> * lnpsi0,
  thrust::complex<FloatType> * htilda)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    htilda[idx] += hfield*thrust::exp(lnpsi1[idx]-lnpsi0[idx]);
    idx += nstep;
  }
}
} // namespace gpu_kernel
