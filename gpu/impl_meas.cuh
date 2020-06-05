// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <typename TraitsClass>
Sampler4SpinHalf<TraitsClass>::Sampler4SpinHalf(AnsatzType & psi, const unsigned long seedNumber, const unsigned long seedDistance):
  BaseParallelSampler<Sampler4SpinHalf, TraitsClass>(psi.get_nInputs(), psi.get_nChains(), seedNumber, seedDistance),
  psi_(psi),
  list_(psi.get_nInputs())
{
  // initialize an index link (for searching a lattice site to flip a spin)
  for (int i=0; i<psi_.get_nInputs(); ++i)
    list_[i].set_item(i);
  int idx0 = 0;
  for (int i=0; i<psi_.get_nInputs(); i++)
  {
    list_[idx0].set_nextptr(&list_[i]);
    idx0 = i;
  }
  list_[idx0].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
}

template <typename TraitsClass>
void Sampler4SpinHalf<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  psi_.initialize(lnpsi_dev);
}

template <typename TraitsClass>
void Sampler4SpinHalf<TraitsClass>::sampling_(thrust::complex<FloatType> * lnpsi_dev)
{
  idxptr_ = idxptr_->next_ptr();
  psi_.forward(idxptr_->get_item(), lnpsi_dev);
}

template <typename TraitsClass>
void Sampler4SpinHalf<TraitsClass>::accept_next_state_(bool * isNewStateAccepted_dev)
{
  psi_.spin_flip(isNewStateAccepted_dev);
}


template <typename TraitsClass>
MeasRenyiEntropy<TraitsClass>::MeasRenyiEntropy(Sampler4SpinHalf<TraitsClass> & smp1, Sampler4SpinHalf<TraitsClass> & smp2, AnsatzType & psi):
  smp1_(smp1),
  smp2_(smp2),
  psi_(psi),
  states3_dev_(psi.get_nChains()*psi.get_nInputs()),
  states4_dev_(psi.get_nChains()*psi.get_nInputs()),
  lnpsi3_dev_(psi.get_nChains()),
  lnpsi4_dev_(psi.get_nChains()),
  kzero(0.0, 0.0)
{}

template <typename TraitsClass>
void MeasRenyiEntropy<TraitsClass>::measure(const int l, const int nIterations, const int nMCSteps, const int nwarmup)
{
  if (l >= psi_.get_nInputs() || l < 0)
    throw std::invalid_argument("l >= psi_.get_nInputs() || l < 0");
  std::cout << "# Now we are in warming up..." << std::flush;
  smp1_.warm_up(nwarmup);
  smp2_.warm_up(nwarmup);
  std::cout << " done." << std::endl << std::flush;
  const int gpuBlockSize1 = CHECK_BLOCK_SIZE(((psi_.get_nInputs()-l)*psi_.get_nChains())),
    gpuBlockSize2 = CHECK_BLOCK_SIZE(psi_.get_nChains());
  thrust::device_vector<thrust::complex<FloatType>> rho2local(psi_.get_nChains(), kzero);
  std::cout << "# Measuring Renyi entropy... (current/total)" << std::endl << std::flush;
  thrust::complex<FloatType> rho2 = kzero;
  for (int n=0; n<nIterations; ++n)
  {
    std::cout << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nIterations << std::flush;
    smp1_.do_mcmc_steps(nMCSteps);
    smp2_.do_mcmc_steps(nMCSteps);
    // lnpsi1_dev : log(C(n_A,q_B)), lnpsi2_dev : log(C(m_A,p_B))
    const thrust::complex<FloatType> * lnpsi1_dev = smp1_.get_lnpsi(), * lnpsi2_dev = smp2_.get_lnpsi();
    // lnpsi3_dev : log(C(n_A,p_B)), lnpsi4_dev : log(C(m_A,q_B))
    const thrust::complex<FloatType> * states1_dev = smp1_.get_quantumStates(), * states2_dev = smp2_.get_quantumStates();
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(states3_dev_.data()), states1_dev,
      sizeof(thrust::complex<FloatType>)*states3_dev_.size(), cudaMemcpyDeviceToDevice));
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(states4_dev_.data()), states2_dev,
      sizeof(thrust::complex<FloatType>)*states4_dev_.size(), cudaMemcpyDeviceToDevice));
    // |n_A>(x)|q_B> <-SWAP-> |m_A>(x)|p_B>  ==> |n_A>(x)|p_B>, |m_A>(x)|q_B> 
    gpu_kernel::Renyi__SwapStates__<<<gpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(psi_.get_nChains(), psi_.get_nInputs(), l, 
      PTR_FROM_THRUST(states3_dev_.data()), PTR_FROM_THRUST(states4_dev_.data()));
    psi_.forward(PTR_FROM_THRUST(states3_dev_.data()), PTR_FROM_THRUST(lnpsi3_dev_.data()), false);
    psi_.forward(PTR_FROM_THRUST(states4_dev_.data()), PTR_FROM_THRUST(lnpsi4_dev_.data()), false);
    // rho2local = (\frac{C(n_A,p_B)*C(m_A,q_B)}{C(n_A,q_B)*C(m_A,p_B)})^*
    gpu_kernel::Renyi__GetRho2local__<<<gpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(lnpsi1_dev, lnpsi2_dev,
      PTR_FROM_THRUST(lnpsi3_dev_.data()), PTR_FROM_THRUST(lnpsi4_dev_.data()), psi_.get_nChains(), PTR_FROM_THRUST(rho2local.data()));
    rho2 = thrust::reduce(thrust::device, rho2local.begin(), rho2local.end(), rho2);
  }
  std::cout << std::endl;
  rho2 /= static_cast<FloatType>((nIterations*psi_.get_nChains()));
  // S_2 = -log(rho2)
  const FloatType S_2 = -1.0*std::log(rho2.real());
  std::cout << "# Renyi entropy(-log(Tr[rho^2])) : " << S_2 << std::endl;
}

namespace gpu_kernel
{
template <typename FloatType>
__global__ void Renyi__SwapStates__(
  const int nChains,
  const int nInputs,
  const int l,
  thrust::complex<FloatType> * states1,
  thrust::complex<FloatType> * states2)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned nstep = gridDim.x*blockDim.x;
  thrust::complex<FloatType> tmp;
  const int ncols = nInputs-l;
  while (idx < nChains*ncols)
  {
    const int k = idx/ncols, i = idx-k*ncols+l;
    tmp = states1[k*nInputs+i];
    states1[k*nInputs+i] = states2[k*nInputs+i];
    states2[k*nInputs+i] = tmp;
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void Renyi__GetRho2local__(
  const thrust::complex<FloatType> * lnpsi1,
  const thrust::complex<FloatType> * lnpsi2,
  const thrust::complex<FloatType> * lnpsi3,
  const thrust::complex<FloatType> * lnpsi4,
  const int nChains,
  thrust::complex<FloatType> * rho2local)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned int nstep = gridDim.x*blockDim.x;
  while (idx < nChains)
  {
    rho2local[idx] = thrust::conj(thrust::exp(lnpsi3[idx]+lnpsi4[idx]-(lnpsi1[idx]+lnpsi2[idx])));
    idx += nstep;
  } 
}
} // namespace gpu_kernel


template <typename TraitsClass>
MeasOverlapIntegral<TraitsClass>::MeasOverlapIntegral(Sampler4SpinHalf<TraitsClass> & smp1, AnsatzType2 & m2):
  smp1_(smp1),
  m2_(m2),
  lnpsi2_dev_(smp1.get_nChains()),
  knInputs(smp1.get_nInputs()),
  knChains(smp1.get_nChains()),
  kgpuBlockSize(CHECK_BLOCK_SIZE(1+(smp1.get_nChains()-1)/NUM_THREADS_PER_BLOCK)),
  kzero(0, 0)
{
  if (smp1.get_nInputs() != m2.get_nInputs())
    throw std::length_error("Check the number of input nodes for each machine");
  if (smp1.get_nChains() != m2.get_nChains())
    throw std::length_error("Check the number of random number sequences for each machine");
}

template <typename TraitsClass>
const thrust::complex<typename TraitsClass::FloatType> MeasOverlapIntegral<TraitsClass>::get_overlapIntegral(const int nTrials,
  const int nwarms, const int nMCSteps, const bool printStatics)
{
  std::cout << "# Now we are in warming up..." << std::flush;
  thrust::host_vector<thrust::complex<FloatType>> ovl(nTrials, kzero);
  smp1_.warm_up(nwarms);
  std::cout << " done." << std::endl << std::flush;
  std::cout << "# Measuring overlap integrals... (current/total)" << std::endl << std::flush;
  thrust::device_vector<thrust::complex<FloatType>> psi2Overpsi0_dev(knChains, kzero);
  for (int n=0; n<nTrials; ++n)
  {
    std::cout << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nTrials << std::flush;
    smp1_.do_mcmc_steps(nMCSteps);
    m2_.forward(smp1_.get_quantumStates(), PTR_FROM_THRUST(lnpsi2_dev_.data()), false);
    gpu_kernel::meas__Psi2OverPsi0__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(smp1_.get_lnpsi(),
      PTR_FROM_THRUST(lnpsi2_dev_.data()), knChains, PTR_FROM_THRUST(psi2Overpsi0_dev.data()));
    ovl[n] = thrust::reduce(thrust::device, psi2Overpsi0_dev.begin(), psi2Overpsi0_dev.end(), kzero)/static_cast<FloatType>(knChains);
  }
  std::cout << std::endl;
  const thrust::complex<FloatType> ovlavg = std::accumulate(ovl.begin(), ovl.end(), kzero)/static_cast<FloatType>(nTrials);
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

namespace gpu_kernel
{
template <typename FloatType>
__global__ void meas__Psi2OverPsi0__(
  const thrust::complex<FloatType> * lnpsi0,
  const thrust::complex<FloatType> * lnpsi2,
  const int nChains,
  thrust::complex<FloatType> * psi2Overpsi0)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned int nstep = gridDim.x*blockDim.x;
  while (idx < nChains)
  {
    psi2Overpsi0[idx] = thrust::exp(lnpsi2[idx]-lnpsi0[idx]);
    idx += nstep;
  }
}
} // namespace gpu_kernel