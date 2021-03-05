// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

namespace spinhalf
{
template <typename TraitsClass>
TFIChain<TraitsClass>::TFIChain(AnsatzType & machine, const int L, const FloatType h, const FloatType J,
  const unsigned long seedNumber, const unsigned long seedDistance, const std::string prefix):
  BaseParallelSampler<TFIChain, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedNumber, seedDistance),
  kL(L),
  knChains(machine.get_nChains()),
  kgpuBlockSize(1+(machine.get_nChains()-1)/NUM_THREADS_PER_BLOCK),
  kh(h),
  kzero(0.0),
  ktwo(2.0),
  machine_(machine),
  list_(machine.get_nInputs()),
  diag_dev_(machine.get_nChains()),
  nnidx_dev_(2*machine.get_nInputs()),
  kJmatrix_dev(2*machine.get_nInputs(), J),
  kprefix(prefix)
{
  if (kL != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L!");
  thrust::host_vector<int> nnidx_host(2*kL); 
  for (int i=0; i<kL; ++i)
  {
    nnidx_host[2*i+0] = (i!=0) ? i-1 : kL-1;
    nnidx_host[2*i+1] = (i!=kL-1) ? i+1 : 0;
  }
  nnidx_dev_ = nnidx_host;
  // Checkerboard link(To implement the MCMC update rule)
  for (int i=0; i<kL; ++i)
    list_[i].set_item(i);
  // black board: even number site
  int idx0 = 0;
  for (int i=0; i<kL; i+=2)
    {
      list_[idx0].set_nextptr(&list_[i]);
      idx0 = i;
    }
  // white board: odd number site
  for (int i=1; i<kL; i+=2)
    {
      list_[idx0].set_nextptr(&list_[i]);
      idx0 = i;
    }
  list_[idx0].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.initialize(lnpsi_dev);
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  // diag_ = \sum_i spin_i*spin_{i+1}
  gpu_kernel::TFI__GetDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, kL,
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(kJmatrix_dev.data()), 2, PTR_FROM_THRUST(diag_dev_.data()));
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::sampling_(thrust::complex<FloatType> * lnpsi_dev)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi_dev);
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::accept_next_state_(bool * isNewStateAccepted_dev)
{
  const int idx = idxptr_->get_item();
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  gpu_kernel::TFI__UpdateDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, kL,
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(kJmatrix_dev.data()), 2, isNewStateAccepted_dev,
    idx, PTR_FROM_THRUST(diag_dev_.data()));
  machine_.spin_flip(isNewStateAccepted_dev);
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::get_htilda_(const thrust::complex<FloatType> * lnpsi0_dev, thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev)
{
  /*
     htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
      --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}
   */
  gpu_kernel::common__copyFromRealToImag__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(diag_dev_.data()), knChains, htilda_dev);
  for (int i=0; i<kL; ++i)
  {
    machine_.forward(i, lnpsi1_dev);
    gpu_kernel::TFI__GetOffDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, kh, lnpsi1_dev, lnpsi0_dev, htilda_dev);
  }
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  machine_.backward(lnpsiGradients_dev);
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate)
{
  machine_.update_variables(trueGradients_dev, learningRate);
}

template <typename TraitsClass>
void TFIChain<TraitsClass>::save_() const
{
  machine_.save(kprefix);
}


// long-range spin-spin interaction with periodic boundary condition
// => 1/2*\sum_{a,b} (s_a*J(a-b)*s_b)
template <typename TraitsClass>
LITFIChain<TraitsClass>::LITFIChain(AnsatzType & machine, const int L, const FloatType h,
  const FloatType J, const double alpha, const bool isPBC,
  const unsigned long seedNumber, const unsigned long seedDistance, const std::string prefix):
  BaseParallelSampler<LITFIChain, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedNumber, seedDistance),
  kL(L),
  knChains(machine.get_nChains()),
  kgpuBlockSize(1+(machine.get_nChains()-1)/NUM_THREADS_PER_BLOCK),
  kh(h),
  kJ(J),
  kzero(0.0),
  kone(1.0),
  machine_(machine),
  list_(L),
  Jmatrix_dev_(L*L),
  SJ_dev_(machine.get_nChains()*L),
  kprefix(prefix)
{
  if (kL != machine.get_nInputs())
    throw std::length_error("machine.get_nInputs() is not the same as L!");
  thrust::host_vector<thrust::complex<FloatType> > Jmatrix_host(kL*kL);
  // periodic boundary condition: Phys. Rev. Lett. 113, 156402 (2014)
  if (isPBC)
  {
    if (kL%2 == 1)
      throw std::invalid_argument("kL%2 == 1 (set \"isPBC\" to \"false\".)");
    for (int i=0; i<kL; ++i)
      for (int j=i+1; j<kL; ++j)
      {
        const FloatType dist = (((j-i)<kL/2) ? (j-i) : kL-(j-i));
        Jmatrix_host[i*kL+j] = J*std::pow(dist, -alpha);
        Jmatrix_host[j*kL+i] = Jmatrix_host[i*kL+j];
      }
  }
  else
  {
    for (int i=0; i<kL; ++i)
      for (int j=i+1; j<kL; ++j)
      {
        const FloatType dist = (j-i);
        Jmatrix_host[i*kL+j] = J*std::pow(dist, -alpha);
        Jmatrix_host[j*kL+i] = Jmatrix_host[i*kL+j];
      }
  }
  Jmatrix_dev_ = Jmatrix_host;

  // Checkerboard link(To implement the MCMC update rule)
  for (int i=0; i<kL; ++i)
    list_[i].set_item(i);
  // black board: even number site
  int idx0 = 0;
  for (int i=0; i<kL; i+=2)
  {
    list_[idx0].set_nextptr(&list_[i]);
    idx0 = i;
  }
  // white board: odd number site
  for (int i=1; i<kL; i+=2)
  {
    list_[idx0].set_nextptr(&list_[i]);
    idx0 = i;
  }
  list_[idx0].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];

  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
}

template <typename TraitsClass>
LITFIChain<TraitsClass>::~LITFIChain()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename TraitsClass>
void LITFIChain<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  thrust::host_vector<thrust::complex<FloatType> > spinStates_host(kL*knChains, kone);
  // Neel order as a starting point of spin states.
  if (kJ > 0)
  {
    for (int k=0; k<knChains; ++k)
      for (int i=0; i<kL; ++i)
        spinStates_host[k*kL+i] = ((i%2 == 0) ? kone : -kone);
  }
  thrust::device_vector<thrust::complex<FloatType> > tmpspinStates_dev(spinStates_host);
  machine_.initialize(lnpsi_dev, PTR_FROM_THRUST(tmpspinStates_dev.data()));
}

template <typename TraitsClass>
void LITFIChain<TraitsClass>::sampling_(thrust::complex<FloatType> * lnpsi_dev)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi_dev);
}

template <typename TraitsClass>
void LITFIChain<TraitsClass>::accept_next_state_(bool * isNewStateAccepted_dev)
{
  const int idx = idxptr_->get_item();
  machine_.spin_flip(isNewStateAccepted_dev);
}

template <typename TraitsClass>
void LITFIChain<TraitsClass>::get_htilda_(const thrust::complex<FloatType> * lnpsi0_dev, thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev)
{
  // htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
  //  --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}

  // htilda <= \sum_{i,j} (s_i J_{i,j} s_j) : spin-spin interaction
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  cublas::gemm(theCublasHandle_, kL, knChains, kL, kone, kzero,
    PTR_FROM_THRUST(Jmatrix_dev_.data()), spinStates_dev, PTR_FROM_THRUST(SJ_dev_.data()));
  gpu_kernel::LITFI__GetDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(SJ_dev_.data()),
    spinStates_dev, knChains, kL, htilda_dev);

  // transverse-field interaction
  for (int i=0; i<kL; ++i)
  {
    machine_.forward(i, lnpsi1_dev);
    gpu_kernel::TFI__GetOffDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, kh, lnpsi1_dev, lnpsi0_dev, htilda_dev);
  }

  gpu_kernel::common__ScalingVector__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(1.0/kL, knChains, htilda_dev);
}

template <typename TraitsClass>
void LITFIChain<TraitsClass>::get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  machine_.backward(lnpsiGradients_dev);
}

template <typename TraitsClass>
void LITFIChain<TraitsClass>::evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate)
{
  machine_.update_variables(trueGradients_dev, learningRate);
}

template <typename TraitsClass>
void LITFIChain<TraitsClass>::save_() const
{
  machine_.save(kprefix);
}


template <typename TraitsClass>
TFISQ<TraitsClass>::TFISQ(AnsatzType & machine, const int L, const FloatType h, const FloatType J,
  const unsigned long seedNumber, const unsigned long seedDistance, const std::string prefix):
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
  kprefix(prefix)
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
void TFISQ<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.initialize(lnpsi_dev);
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  // diag_ = \sum_i spin_i*spin_{i+1}
  gpu_kernel::TFI__GetDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL),
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(kJmatrix_dev.data()), 4, PTR_FROM_THRUST(diag_dev_.data()));
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
  const int idx = idxptr_->get_item();
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  gpu_kernel::TFI__UpdateDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL),
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(kJmatrix_dev.data()), 4, isNewStateAccepted_dev,
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
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, lnpsi1_dev);
    gpu_kernel::TFI__GetOffDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, kh, lnpsi1_dev, lnpsi0_dev, htilda_dev);
  }
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  machine_.backward(lnpsiGradients_dev);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate)
{
  machine_.update_variables(trueGradients_dev, learningRate);
}

template <typename TraitsClass>
void TFISQ<TraitsClass>::save_() const
{
  machine_.save(kprefix);
}


template <typename TraitsClass>
TFICheckerBoard<TraitsClass>::TFICheckerBoard(AnsatzType & machine, const int L, const FloatType h,
  const std::array<FloatType, 2> J1_J2, const bool isPeriodicBoundary, const unsigned long seedNumber,
  const unsigned long seedDistance, const std::string prefix):
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
  kprefix(prefix)
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
  thrust::host_vector<int> nnidx_host(8*kL*kL);
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      const int idx = i*kL+j;
      Jmatrix_host[8*idx+0] = (i == 0) ? (kJ1*isPeriodicBoundary) : kJ1; // up
      Jmatrix_host[8*idx+1] = (i == kL-1) ? (kJ1*isPeriodicBoundary) : kJ1; // down
      Jmatrix_host[8*idx+2] = (j == 0) ? (kJ1*isPeriodicBoundary) : kJ1; // left
      Jmatrix_host[8*idx+3] = (j == kL-1) ? (kJ1*isPeriodicBoundary) : kJ1; // right
      if ((i+j)%2 == 0)
      {
        Jmatrix_host[8*idx+4] = (i == 0 || j == kL-1) ? (kJ2*isPeriodicBoundary) : kJ2; // up-right
        Jmatrix_host[8*idx+5] = (i == kL-1 || j == 0) ? (kJ2*isPeriodicBoundary) : kJ2; // down-left
      }
      else
      {
        Jmatrix_host[8*idx+6] = (i == 0 || j == 0) ? (kJ2*isPeriodicBoundary) : kJ2; // up-left
        Jmatrix_host[8*idx+7] = (i == kL-1 || j == kL-1) ? (kJ2*isPeriodicBoundary) : kJ2; // down-right
      }
    }
  Jmatrix_dev_ = Jmatrix_host;
  // table of the index for the nearest-neighbor sites
  for (int i=0; i<kL; ++i)
    for (int j=0; j<kL; ++j)
    {
      const int idx = i*kL+j;
      nnidx_host[8*idx+0] = (i == 0) ? (kL-1)*kL+j : (i-1)*kL+j; // up
      nnidx_host[8*idx+1] = (i == kL-1) ? j : (i+1)*kL+j; // down
      nnidx_host[8*idx+2] = (j == 0) ? i*kL+kL-1 : i*kL+j-1; // left
      nnidx_host[8*idx+3] = (j == kL-1) ? i*kL : i*kL+j+1; // right
    }
  for (int i=1; i<kL-1; ++i)
    for (int j=1; j<kL-1; ++j)
    {
      const int idx = i*kL+j;
      nnidx_host[8*idx+4] = (i-1)*kL+j+1; // up-right
      nnidx_host[8*idx+5] = (i+1)*kL+j-1; // down-left
      nnidx_host[8*idx+6] = (i-1)*kL+j-1; // up-left
      nnidx_host[8*idx+7] = (i+1)*kL+j+1; // down-right
    }
  // i == 0, 1 <= j <= kL-2
  for (int j=1; j<kL-1; ++j)
  {
    const int idx = j;
    nnidx_host[8*idx+4] = (kL-1)*kL+j+1; // up-right
    nnidx_host[8*idx+5] = kL+j-1; // down-left
    nnidx_host[8*idx+6] = (kL-1)*kL+j-1; // up-left
    nnidx_host[8*idx+7] = kL+j+1; // down-right
  }
  // i == kL-1, 1 <= j <= kL-2
  for (int j=1; j<kL-1; ++j)
  {
    const int idx = (kL-1)*kL+j;
    nnidx_host[8*idx+4] = (kL-2)*kL+j+1; // up-right
    nnidx_host[8*idx+5] = j-1; // down-left
    nnidx_host[8*idx+6] = (kL-2)*kL+j-1; // up-left
    nnidx_host[8*idx+7] = j+1; // down-right
  }
  // 1 <= i <= kL-2, j == 0
  for (int i=1; i<kL-1; ++i)
  {
    const int idx = i*kL;
    nnidx_host[8*idx+4] = (i-1)*kL+1; // up-right
    nnidx_host[8*idx+5] = (i+1)*kL+kL-1; // down-left
    nnidx_host[8*idx+6] = (i-1)*kL+kL-1; // up-left
    nnidx_host[8*idx+7] = (i+1)*kL+1; // down-right
  }
  // 1 <= i <= kL-2, j == kL-1
  for (int i=1; i<kL-1; ++i)
  {
    const int idx = i*kL+kL-1;
    nnidx_host[8*idx+4] = (i-1)*kL; // up-right
    nnidx_host[8*idx+5] = (i+1)*kL+kL-2; // down-left
    nnidx_host[8*idx+6] = (i-1)*kL+kL-2; // up-left
    nnidx_host[8*idx+7] = (i+1)*kL; // down-right
  }
  // i == 0, j == 0
  nnidx_host[4] = (kL-1)*kL+1; // up-right
  nnidx_host[5] = kL+kL-1; // down-left
  nnidx_host[6] = (kL-1)*kL+kL-1; // up-left
  nnidx_host[7] = kL+1; // down-right
  // i == 0, j == kL-1
  nnidx_host[8*(kL-1)+4] = (kL-1)*kL; // up-right
  nnidx_host[8*(kL-1)+5] = kL+kL-2; // down-left
  nnidx_host[8*(kL-1)+6] = (kL-1)*kL+kL-2; // up-left
  nnidx_host[8*(kL-1)+7] = kL; // down-right
  // i == kL-1, j == 0
  nnidx_host[8*(kL-1)*kL+4] = (kL-2)*kL+1; // up-right
  nnidx_host[8*(kL-1)*kL+5] = kL-1; // down-left
  nnidx_host[8*(kL-1)*kL+6] = (kL-2)*kL+kL-1; // up-left
  nnidx_host[8*(kL-1)*kL+7] = 1; // down-right
  // i == kL-1, j == kL-1
  nnidx_host[8*((kL-1)*kL+kL-1)+4] = (kL-2)*kL; // up-right
  nnidx_host[8*((kL-1)*kL+kL-1)+5] = kL-2; // down-left
  nnidx_host[8*((kL-1)*kL+kL-1)+6] = (kL-2)*kL+kL-2; // up-left
  nnidx_host[8*((kL-1)*kL+kL-1)+7] = 0; // down-right
  nnidx_dev_ = nnidx_host;
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
void TFICheckerBoard<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.initialize(lnpsi_dev);
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  // diag_ = \sum_i spin_i*spin_{i+1}
  gpu_kernel::TFI__GetDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL),
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(Jmatrix_dev_.data()), 8, PTR_FROM_THRUST(diag_dev_.data()));
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
  const int idx = idxptr_->get_item();
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  gpu_kernel::TFI__UpdateDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knChains, (kL*kL),
    PTR_FROM_THRUST(nnidx_dev_.data()), PTR_FROM_THRUST(Jmatrix_dev_.data()), 8, isNewStateAccepted_dev,
    idx, PTR_FROM_THRUST(diag_dev_.data()));
  machine_.spin_flip(isNewStateAccepted_dev);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::get_htilda_(const thrust::complex<FloatType> * lnpsi0_dev, thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev)
{
  //   htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
  //    --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}
  gpu_kernel::common__copyFromRealToImag__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(diag_dev_.data()), knChains, htilda_dev);
  for (int i=0; i<knSites; ++i)
  {
    machine_.forward(i, lnpsi1_dev);
    gpu_kernel::TFI__GetOffDiagElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, kh, lnpsi1_dev, lnpsi0_dev, htilda_dev);
  }
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  machine_.backward(lnpsiGradients_dev);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate)
{
  machine_.update_variables(trueGradients_dev, learningRate);
}

template <typename TraitsClass>
void TFICheckerBoard<TraitsClass>::save_() const
{
  machine_.save(kprefix);
}
} // namespace spinhalf

namespace fermion 
{
namespace jordanwigner
{
template <typename TraitsClass>
HubbardChain<TraitsClass>::HubbardChain(AnsatzType & machine, const FloatType U,
  const FloatType t, const std::vector<FloatType> & V, const std::array<int, 2> & np,
  const bool usePBC, const unsigned long seedNumber,
  const unsigned long seedDistance, const std::string prefix, const bool useSpinStates):
  BaseParallelSampler<HubbardChain, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedNumber, seedDistance),
  machine_(machine),
  knSites(machine.get_nInputs()/2),
  knChains(machine.get_nChains()),
  kgpuBlockSize(1+(machine.get_nChains()-1)/NUM_THREADS_PER_BLOCK),
  np_(np),
  kusePBC(usePBC),
  kuseSpinStates(useSpinStates),
  kU(U),
  kt(t),
  kzero(0),
  ktwo(2),
  exchanger_(machine.get_nChains(), machine.get_nInputs(), seedNumber*12345ul, seedDistance),
  spinPairIdx_dev_(machine.get_nChains()),
  tmpspinPairIdx_dev_(machine.get_nChains()),
  V_dev_(V.begin(), V.end()),
  spinStates_host_(machine.get_nChains()*machine.get_nInputs()),
  kprefix(prefix)
{
  // ranges of machine inputs:
  // [0~knSites) -> spin up; [knSites~2*knSites) -> spin down
  if (machine.get_nInputs()%2 != 0)
    throw std::invalid_argument("machine.get_nInputs()%2 != 0");
  if (V.size() != machine.get_nInputs())
    throw std::invalid_argument("V.size() != machine.get_nInputs()");
}

template <typename TraitsClass>
void HubbardChain<TraitsClass>::get_htilda_(const thrust::complex<FloatType> * lnpsi0_dev,
  thrust::complex<FloatType> * lnpsi1_dev, thrust::complex<FloatType> * htilda_dev)
{
  const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
  // hopping term
  gpu_kernel::common__SetValues__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(htilda_dev, knChains, kzero);
  // s : 0 (spin up), 1 (spin down)
  for (int s=0; s<2; ++s)
  {
    // left to right direction
    for (int i=0; i<knSites-1; ++i)
    {
      thrust::fill(tmpspinPairIdx_dev_.begin(), tmpspinPairIdx_dev_.end(), thrust::pair<int, int>(s*knSites+i, s*knSites+i+1));
      machine_.forward(tmpspinPairIdx_dev_, lnpsi1_dev);
      gpu_kernel::HubbardChain__AddedHoppingElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains,
        knSites, spinStates_dev, PTR_FROM_THRUST(tmpspinPairIdx_dev_.data()), lnpsi0_dev, lnpsi1_dev, htilda_dev);
    }
    // right to left direction
    for (int i=1; i<knSites; ++i)
    {
      thrust::fill(tmpspinPairIdx_dev_.begin(), tmpspinPairIdx_dev_.end(), thrust::pair<int, int>(s*knSites+i, s*knSites+i-1));
      machine_.forward(tmpspinPairIdx_dev_, lnpsi1_dev);
      gpu_kernel::HubbardChain__AddedHoppingElem__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains,
        knSites, spinStates_dev, PTR_FROM_THRUST(tmpspinPairIdx_dev_.data()), lnpsi0_dev, lnpsi1_dev, htilda_dev);
    }
    // edge to edge
    if (!kusePBC)
      continue;
    thrust::fill(tmpspinPairIdx_dev_.begin(), tmpspinPairIdx_dev_.end(), thrust::pair<int, int>(s*knSites, s*knSites+knSites-1));
    machine_.forward(tmpspinPairIdx_dev_, lnpsi1_dev);
    gpu_kernel::HubbardChain__AddedHoppingElemEdge__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(s, knChains, knSites, spinStates_dev,
      PTR_FROM_THRUST(tmpspinPairIdx_dev_.data()), lnpsi0_dev, lnpsi1_dev, htilda_dev);
  }
  gpu_kernel::common__ScalingVector__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(-0.25*kt, knChains, htilda_dev);
  // onsite interaction term
  gpu_kernel::HubbardChain__AddedOnSiteInteraction__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, knSites, kU, spinStates_dev, htilda_dev);
  // onsite potential trap
  gpu_kernel::HubbardChain__AddedPotentialTrap__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, knSites,
    PTR_FROM_THRUST(V_dev_.data()), spinStates_dev, htilda_dev);

  gpu_kernel::common__ScalingVector__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(1.0/knSites, knChains, htilda_dev);
}

template <typename TraitsClass>
void HubbardChain<TraitsClass>::get_lnpsiGradients_(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  machine_.backward(lnpsiGradients_dev);
}

template <typename TraitsClass>
void HubbardChain<TraitsClass>::evolve_(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate)
{
  machine_.update_variables(trueGradients_dev, learningRate);
}

template <typename TraitsClass>
void HubbardChain<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  // +1 : particle is filling at the site.
  // -1 : particle is empty at the site.
  const int nInputs = 2*knSites;
  thrust::host_vector<thrust::complex<FloatType> > spinStates_host(knChains*nInputs);
  if (kuseSpinStates)
  {
    try
    {
      this->load_spin_data_(spinStates_host);
    }
    catch (const std::string & warning)
    {
      std::cout << warning << std::endl;
      this->initialize_spins_randomly_(spinStates_host);
    } 
  }
  else
    this->initialize_spins_randomly_(spinStates_host);
  thrust::device_vector<thrust::complex<FloatType> > spinStates_dev(spinStates_host);
  machine_.initialize(lnpsi_dev, PTR_FROM_THRUST(spinStates_dev.data()));
  // initialize spin-exchange sampler
  exchanger_.init(kawasaki::IsBondState(), spinStates_dev.data());
  exchanger_.get_indexes_of_spin_pairs(spinPairIdx_dev_);
}

template <typename TraitsClass>
void HubbardChain<TraitsClass>::load_spin_data_(thrust::host_vector<thrust::complex<FloatType>> & spinStates_host)
{
  assert(spinStates_host.size() == knChains*2*knSites);
  const std::string filename = kprefix + "Ds.dat";
  std::ifstream rfile(filename);
  if (!(rfile.is_open()))
  {
    const std::string warning = "# WARNING: " + filename + " is not exist...";
    throw warning;
  }
  // read data
  for (auto & item : spinStates_host)
    if (!(rfile >> item))
    {
      const std::string warning = "# WARNING: spinStates_host.size() < knChain*2*knSites";
      throw warning;
    }
  // check # of particles
  for (int k=0; k<knChains; ++k)
  {
    // s : 0 (spin up), 1 (spin down)
    for (int s=0; s<2; ++s)
    {
      int sum = 0;
      for (int i=0; i<knSites; ++i)
        sum += static_cast<int>(spinStates_host[k*2*knSites+s*knSites+i].real());
      if (sum != 2*np_[s]-knSites)
      {
        const std::string warning = "# WARNING: The # of particles is not same as np[s] (k: "
          + std::to_string(k) + ", s:" + std::to_string(s) + ").";
        throw warning;
      }
    }
  }
  rfile.close();
}

template <typename TraitsClass>
void HubbardChain<TraitsClass>::initialize_spins_randomly_(thrust::host_vector<thrust::complex<FloatType>> & spinStates_host)
{
  assert(spinStates_host.size() == knChains*2*knSites);
  thrust::fill(spinStates_host.begin(), spinStates_host.end(), -1.0);
  std::vector<int> idx(knSites);
  const thrust::complex<FloatType> one(1);
  for (int i=0; i<idx.size(); ++i)
    idx[i] = i;
  for (int k=0; k<knChains; ++k)
  {
    // s : 0 (spin up), 1 (spin down)
    for (int s=0; s<2; ++s)
    {
      std::shuffle(idx.begin(), idx.end(), std::default_random_engine((12345u*k+9876543210u*s)));
      for (int n=0; n<np_[s]; ++n)
        spinStates_host[k*2*knSites+s*knSites+idx[n]] = one;
    }
  }
  std::cout << "# spin states are randomly initialized..." << std::endl << std::flush;
}

template <typename TraitsClass>
void HubbardChain<TraitsClass>::sampling_(thrust::complex<FloatType> * lnpsi_dev)
{
  exchanger_.get_indexes_of_spin_pairs(spinPairIdx_dev_);
  machine_.forward(spinPairIdx_dev_, lnpsi_dev);
}

template <typename TraitsClass>
void HubbardChain<TraitsClass>::accept_next_state_(bool * isNewStateAccepted_dev)
{
  machine_.spin_flip(isNewStateAccepted_dev, spinPairIdx_dev_);
  exchanger_.do_exchange(isNewStateAccepted_dev);
}

template <typename TraitsClass>
void HubbardChain<TraitsClass>::save_() const
{
  machine_.save(kprefix);
  // save spin states
  CHECK_ERROR(cudaSuccess, cudaMemcpy((void*)spinStates_host_.data(), (void*)machine_.get_spinStates(),
    sizeof(thrust::complex<FloatType>)*spinStates_host_.size(), cudaMemcpyDeviceToHost));
  const std::string filename = kprefix + "Ds.dat";
  std::ofstream wfile(filename);
  for (int k=0; k<machine_.get_nChains(); ++k)
  {
    for (int i=0; i<machine_.get_nInputs(); ++i)
      wfile << static_cast<int>(spinStates_host_[k*machine_.get_nInputs()+i].real()) << " ";
    wfile << std::endl;
  }
  wfile.close();
}
} // end namespace fermion
} // end namespace jordanwigner 

namespace gpu_kernel
{
template <typename FloatType>
__global__ void TFI__GetDiagElem__(const thrust::complex<FloatType> * spinStates, const int nChains,
  const int nSites, const int * nnidx, const FloatType * Jmatrix, const int nnn, FloatType * diag)
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
      FloatType sumNNInteractions = zero;
      for (int n=0; n<nnn; ++n)
        sumNNInteractions += spinStates[kstate+nnidx[nnn*i+n]].real()*Jmatrix[nnn*i+n];
      diag[idx] += spinStates[kstate+i].real()*sumNNInteractions;
    }
    diag[idx] *= 0.5;
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void TFI__UpdateDiagElem__(const thrust::complex<FloatType> * spinStates, const int nChains,
  const int nSites, const int * nnidx, const FloatType * Jmatrix, const int nnn,
  const bool * updateList, const int siteIdx, FloatType * diag)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType twoDelta[2] = {0.0, 2.0}, zero = 0.0;
  while (idx < nChains)
  {
    const int kstate = idx*nSites;
    FloatType sumNNInteractions = zero;
    for (int n=0; n<nnn; ++n)
      sumNNInteractions += spinStates[kstate+nnidx[nnn*siteIdx+n]].real()*Jmatrix[nnn*siteIdx+n];
    diag[idx] -= twoDelta[updateList[idx]]*spinStates[kstate+siteIdx].real()*sumNNInteractions;
    idx += nstep;
  }
}

// htilda[k] += hfield*exp(lnpsi1[k] - lnpsi0[k]);
template <typename FloatType>
__global__ void TFI__GetOffDiagElem__(const int nChains, const FloatType hfield, const thrust::complex<FloatType> * lnpsi1,
  const thrust::complex<FloatType> * lnpsi0, thrust::complex<FloatType> * htilda)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    htilda[idx] += hfield*thrust::exp(lnpsi1[idx]-lnpsi0[idx]);
    idx += nstep;
  }
}

// return \sum_{i,j} 1/2*(s_i \tilde{J}_{i,j} s_j)
template <typename FloatType>
__global__ void LITFI__GetDiagElem__(const thrust::complex<FloatType> * SJ, const thrust::complex<FloatType> * spinStates,
  const int nChains, const int nSites, thrust::complex<FloatType> * htilda)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType zero = 0, half = 0.5;
  while (idx < nChains)
  {
    htilda[idx] = zero;
    for (int i=0; i<nSites; ++i)
      htilda[idx] += SJ[nSites*idx+i].real()*spinStates[nSites*idx+i].real();
    htilda[idx] = half*htilda[idx];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void HubbardChain__AddedHoppingElem__(const int nChains, const int nSites,
  const thrust::complex<FloatType> * spinStates, const thrust::pair<int, int> * spinPairIdx,
  const thrust::complex<FloatType> * lnpsi0, const thrust::complex<FloatType> * lnpsi1, thrust::complex<FloatType> * htilda)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType one = 1;
  while (idx < nChains)
  {
    const int k = idx;
    const int spinFlipIdx1 = spinPairIdx[k].first,
              spinFlipIdx2 = spinPairIdx[k].second;
    htilda[k] += (one+spinStates[k*2*nSites+spinFlipIdx1].real())*
                 (one-spinStates[k*2*nSites+spinFlipIdx2].real())*
                 thrust::exp(lnpsi1[k]-lnpsi0[k]);
    idx += nstep;
  }
}

// flavor : 0(spin up) or 1(spin down)
template <typename FloatType>
__global__ void HubbardChain__AddedHoppingElemEdge__(const int flavor, const int nChains, const int nSites,
  const thrust::complex<FloatType> * spinStates, const thrust::pair<int, int> * spinPairIdx,
  const thrust::complex<FloatType> * lnpsi0, const thrust::complex<FloatType> * lnpsi1, thrust::complex<FloatType> * htilda)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType one = 1, two = 2;
  while (idx < nChains)
  {
    const int k = idx;
    const int spinFlipIdx1 = spinPairIdx[k].first,
              spinFlipIdx2 = spinPairIdx[k].second;
    FloatType sp = one;
    for (int i=flavor*nSites+1; i<flavor*nSites+nSites-1; ++i)
      sp *= -one*spinStates[k*2*nSites+i].real();
    htilda[k] += two*sp*(one-spinStates[k*2*nSites+spinFlipIdx1].real()*
                        spinStates[k*2*nSites+spinFlipIdx2].real())*
                        thrust::exp(lnpsi1[k]-lnpsi0[k]);
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void HubbardChain__AddedOnSiteInteraction__(const int nChains, const int nSites, const FloatType U,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * htilda)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType one = 1, Uquater = 0.25*U;
  while (idx < nChains)
  {
    const int k = idx;
    for (int i=0; i<nSites; ++i)
      htilda[k] += Uquater*(one+spinStates[k*2*nSites+i].real())*(one+spinStates[k*2*nSites+i+nSites].real());
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void HubbardChain__AddedPotentialTrap__(const int nChains, const int nSites, const FloatType * V,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * htilda)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType one = 1, half = 0.5;
  while (idx < nChains)
  {
    const int k = idx;
    for (int i=0; i<nSites; ++i)
      htilda[k] += half*(V[i]*(one+spinStates[k*2*nSites+i].real())+V[i+nSites]*(one+spinStates[k*2*nSites+i+nSites].real()));
    idx += nstep;
  }
}
} // namespace gpu_kernel
