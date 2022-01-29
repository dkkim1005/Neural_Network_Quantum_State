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
typename TraitsClass::FloatType MeasRenyiEntropy<TraitsClass>::measure(const int l, const int nIterations, const int nMCSteps, const int nwarmup)
{
  if (l >= psi_.get_nInputs() || l < 0)
    throw std::invalid_argument("l >= psi_.get_nInputs() || l < 0");
  std::cout << "# Now we are in warming up..." << std::flush;
  smp1_.warm_up(nwarmup);
  smp2_.warm_up(nwarmup);
  std::cout << " done." << std::endl << std::flush;
  const int gpuBlockSize1 = CHECK_BLOCK_SIZE(((psi_.get_nInputs()-l)*psi_.get_nChains())),
    gpuBlockSize2 = CHECK_BLOCK_SIZE(psi_.get_nChains());
  thrust::device_vector<thrust::complex<FloatType>> rho2local_dev(psi_.get_nChains(), kzero);
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
    // rho2local_dev = (\frac{C(n_A,p_B)*C(m_A,q_B)}{C(n_A,q_B)*C(m_A,p_B)})^*
    gpu_kernel::meas__GetRho2local__<<<gpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(lnpsi1_dev, lnpsi2_dev,
      PTR_FROM_THRUST(lnpsi3_dev_.data()), PTR_FROM_THRUST(lnpsi4_dev_.data()), psi_.get_nChains(), PTR_FROM_THRUST(rho2local_dev.data()));
    rho2 = thrust::reduce(thrust::device, rho2local_dev.begin(), rho2local_dev.end(), rho2);
  }
  std::cout << std::endl;
  rho2 /= static_cast<FloatType>((nIterations*psi_.get_nChains()));
  // S_2 = -log(rho2)
  const FloatType S_2 = -1.0*std::log(rho2.real());
  std::cout << "# Renyi entropy(-log(Tr[rho^2])) : " << S_2 << std::endl;
  return S_2;
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
__global__ void meas__GetRho2local__(
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


template <typename TraitsClass>
MeasFidelity<TraitsClass>::MeasFidelity(Sampler4SpinHalf<TraitsClass> & smp1, Sampler4SpinHalf<TraitsClass> & smp2, AnsatzType & psi1, AnsatzType & psi2):
  smp1_(smp1),
  smp2_(smp2),
  psi1_(psi1),
  psi2_(psi2),
  lnpsi3_dev_(smp1.get_nChains()),
  lnpsi4_dev_(smp1.get_nChains()),
  knInputs(smp1.get_nInputs()),
  knChains(smp1.get_nChains()),
  kgpuBlockSize(CHECK_BLOCK_SIZE(psi1_.get_nChains())),
  kzero(0, 0)
{}


template <typename TraitsClass>
std::pair<typename TraitsClass::FloatType, typename TraitsClass::FloatType> MeasFidelity<TraitsClass>::measure(const int nMeas,
  const int nwarms, const int nMCSteps)
{
  std::cout << "# Now we are in warming up..." << std::flush;
  smp1_.warm_up(nwarms);
  smp2_.warm_up(nwarms);
  std::cout << " done." << std::endl << std::flush;
  std::cout << "# Measuring fidelity... (current/total)" << std::endl << std::flush;
  thrust::device_vector<thrust::complex<FloatType>> rho2local_dev(knChains, kzero);
  thrust::host_vector<FloatType> rho2_host(nMeas, kzero.real());
  for (int n=0; n<nMeas; ++n)
  {
    std::cout << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nMeas << std::flush;
    smp1_.do_mcmc_steps(nMCSteps);
    smp2_.do_mcmc_steps(nMCSteps);
    // lnpsi1_dev : log(<\sigma_1|\psi_1>), lnpsi2_dev : log(<\sigma_2|\psi_2>)
    const thrust::complex<FloatType> * lnpsi1_dev = smp1_.get_lnpsi(), * lnpsi2_dev = smp2_.get_lnpsi();
    // lnpsi3_dev_ : log(<\sigma_2|\psi_1>)
    psi1_.forward(smp2_.get_quantumStates(), PTR_FROM_THRUST(lnpsi3_dev_.data()), false);
    // lnpsi4_dev_ : log(<\sigma_1|\psi_2>)
    psi2_.forward(smp1_.get_quantumStates(), PTR_FROM_THRUST(lnpsi4_dev_.data()), false);
    // rho2local = (\frac{C(n_A,p_B)*C(m_A,q_B)}{C(n_A,q_B)*C(m_A,p_B)})^*
    gpu_kernel::meas__GetRho2local__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(lnpsi1_dev, lnpsi2_dev,
      PTR_FROM_THRUST(lnpsi3_dev_.data()), PTR_FROM_THRUST(lnpsi4_dev_.data()), knChains, PTR_FROM_THRUST(rho2local_dev.data()));
    rho2_host[n] = thrust::reduce(thrust::device, rho2local_dev.begin(), rho2local_dev.end(), kzero).real()/knChains;
  }
  std::cout << std::endl;
  const FloatType rhoMean = std::sqrt((thrust::reduce(thrust::host, rho2_host.begin(), rho2_host.end(), kzero).real())/nMeas);
  FloatType err = kzero.real();
  for (const auto & item : rho2_host)
    err += std::pow(std::sqrt(item) - rhoMean, 2);
  err = std::sqrt(err/(rho2_host.size()-1)/rho2_host.size());
  return std::pair<typename TraitsClass::FloatType, typename TraitsClass::FloatType>(rhoMean, err);
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


template <typename TraitsClass>
MeasSpinZSpinZCorrelation<TraitsClass>::MeasSpinZSpinZCorrelation(Sampler4SpinHalf<TraitsClass> & smp):
  smp_(smp),
  ss_dev_(smp.get_nInputs()*smp.get_nInputs()),
  knInputs(smp.get_nInputs()),
  knChains(smp.get_nChains()),
  kzero(0, 0),
  kone(1, 0)
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
}

template <typename TraitsClass>
MeasSpinZSpinZCorrelation<TraitsClass>::~MeasSpinZSpinZCorrelation()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename TraitsClass>
void MeasSpinZSpinZCorrelation<TraitsClass>::measure(const int nIterations, const int nMCSteps, const int nwarmup, FloatType * ss


#ifdef __KISTI_GPU__
  , const std::string logpath
#endif


)
{


#ifdef __KISTI_GPU__
  std::ofstream logfile(logpath, std::fstream::app);
#endif



#ifdef __KISTI_GPU__
  logfile << "# Now we are in warming up..." << std::flush;
#else
  std::cout << "# Now we are in warming up..." << std::flush;
#endif


  smp_.warm_up(nwarmup);


#ifdef __KISTI_GPU__
  logfile << " done." << std::endl << std::flush;
  logfile << "# Measuring spin-spin correlation... (current/total)" << std::endl << std::flush;
#else
  std::cout << " done." << std::endl << std::flush;
  std::cout << "# Measuring spin-spin correlation... (current/total)" << std::endl << std::flush;
#endif


  thrust::fill(ss_dev_.begin(), ss_dev_.end(), kzero);
  const FloatType oneOverTotalMeas = 1/static_cast<FloatType>(nIterations*knChains);
  for (int n=0; n<nIterations; ++n)
  {


#ifdef __KISTI_GPU__
    logfile << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nIterations << std::flush;
#else
    std::cout << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nIterations << std::flush;
#endif


    smp_.do_mcmc_steps(nMCSteps);
    cublas::herk(theCublasHandle_, knInputs, knChains, oneOverTotalMeas, smp_.get_quantumStates(), kone.real(), PTR_FROM_THRUST(ss_dev_.data()));
  }


#ifdef __KISTI_GPU__
  logfile << std::endl;
  logfile.close();
#else
  std::cout << std::endl;
#endif


  thrust::host_vector<thrust::complex<FloatType>> ss_host(ss_dev_);
  for (int i=0; i<knInputs; ++i)
    for (int j=0; j<knInputs; ++j)
    {
      const int idx = i*knInputs+j;
      ss[idx] = ss_host[idx].real();
    }
}



namespace gpu_kernel
{
template <typename FloatType>
__global__ void meas__NN2PointsCorr__(
  const thrust::complex<FloatType> * s,
  thrust::complex<FloatType> * nnsz2,
  const int nChains,
  const int nInputs)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned int nstep = gridDim.x*blockDim.x;
  while (idx < nChains*(nInputs-1))
  {
    const int i = idx/nChains, k = idx-i*nChains;
    const int siteIdx = k*nInputs+i;
    // nnsz2_{k,i} = s_{k,i}*s_{k,i+1}
    nnsz2[siteIdx] = s[siteIdx]*s[siteIdx+1];
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    const int siteIdx = (idx+1)*nInputs-1;
    // nnsz2_{k,nInputs-1} = s_{k,nInputs-1}*s_{k,0}
    nnsz2[siteIdx] = s[siteIdx]*s[siteIdx-nInputs+1];
    idx += nstep;
  }
}
} // namespace gpu_kernel


template <typename TraitsClass>
Meas4PointsSpinZCorrelation<TraitsClass>::Meas4PointsSpinZCorrelation(Sampler4SpinHalf<TraitsClass> & smp):
  smp_(smp),
  sz4_dev_(smp.get_nInputs()*smp.get_nInputs()),
  nnsz2_dev_(smp.get_nChains()*smp.get_nInputs()),
  knInputs(smp.get_nInputs()),
  knChains(smp.get_nChains()),
  kgpuBlockSize(CHECK_BLOCK_SIZE(1+(nnsz2_dev_.size()-1)/NUM_THREADS_PER_BLOCK)),
  kzero(0, 0),
  kone(1, 0)
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
}

template <typename TraitsClass>
Meas4PointsSpinZCorrelation<TraitsClass>::~Meas4PointsSpinZCorrelation()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename TraitsClass>
void Meas4PointsSpinZCorrelation<TraitsClass>::measure(const int nIterations, const int nMCSteps,
  const int nwarmup, FloatType * sz4
#ifdef __KISTI_GPU__
  , const std::string logpath
#endif
)
{
#ifdef __KISTI_GPU__
  std::ofstream logfile(logpath, std::fstream::app);
  logfile << "# Now we are in warming up..." << std::flush;
#else
  std::cout << "# Now we are in warming up..." << std::flush;
#endif

  smp_.warm_up(nwarmup);

#ifdef __KISTI_GPU__
  logfile << " done." << std::endl << std::flush;
  logfile << "# Measuring spin-spin correlation... (current/total)" << std::endl << std::flush;
#else
  std::cout << " done." << std::endl << std::flush;
  std::cout << "# Measuring spin-spin correlation... (current/total)" << std::endl << std::flush;
#endif

  thrust::fill(sz4_dev_.begin(), sz4_dev_.end(), kzero);
  const FloatType oneOverTotalMeas = 1/static_cast<FloatType>(nIterations*knChains);
  for (int n=0; n<nIterations; ++n)
  {
#ifdef __KISTI_GPU__
    logfile << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nIterations << std::flush;
#else
    std::cout << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nIterations << std::flush;
#endif
    smp_.do_mcmc_steps(nMCSteps);
    gpu_kernel::meas__NN2PointsCorr__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(smp_.get_quantumStates(),
      PTR_FROM_THRUST(nnsz2_dev_.data()), knChains, knInputs);
    cublas::herk(theCublasHandle_, knInputs, knChains, oneOverTotalMeas, PTR_FROM_THRUST(nnsz2_dev_.data()),
      kone.real(), PTR_FROM_THRUST(sz4_dev_.data()));
  }

#ifdef __KISTI_GPU__
  logfile << std::endl;
  logfile.close();
#else
  std::cout << std::endl;
#endif

  thrust::host_vector<thrust::complex<FloatType>> sz4_host(sz4_dev_);
  for (int i=0; i<knInputs; ++i)
    for (int j=0; j<knInputs; ++j)
    {
      const int idx = i*knInputs+j;
      sz4[idx] = sz4_host[idx].real();
    }
}



template <typename TraitsClass>
MeasSpinXSpinXCorrelation<TraitsClass>::MeasSpinXSpinXCorrelation(Sampler4SpinHalf<TraitsClass> & smp, AnsatzType & psi):
  smp_(smp),
  psi_(psi),
  ss_dev_(smp.get_nInputs()*smp.get_nInputs()),
  s_dev_(smp.get_nInputs()),
  tmplnpsi_dev_(smp.get_nChains()),
  kones(smp.get_nChains(), thrust::complex<FloatType>(1, 0)),
  spinPairFlipIdx_dev_(smp.get_nChains()),
  knInputs(smp.get_nInputs()),
  knChains(smp.get_nChains()),
  kgpuBlockSize(CHECK_BLOCK_SIZE(1+(smp.get_nChains()-1)/NUM_THREADS_PER_BLOCK)),
  kzero(0, 0),
  kone(1, 0)
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
}

template <typename TraitsClass>
MeasSpinXSpinXCorrelation<TraitsClass>::~MeasSpinXSpinXCorrelation()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename TraitsClass>
void MeasSpinXSpinXCorrelation<TraitsClass>::measure(const int nIterations,
  const int nMCSteps, const int nwarmup, FloatType * ss, FloatType * s


#ifdef __KISTI_GPU__
, const std::string logpath
#endif


  )
{


#ifdef __KISTI_GPU__
  std::ofstream logfile(logpath, std::fstream::app);
#endif


  thrust::fill(ss_dev_.begin(), ss_dev_.end(), kzero);
  thrust::fill(s_dev_.begin(), s_dev_.end(), kzero);
  thrust::device_vector<thrust::complex<FloatType>> ss_accum_dev(knInputs*knInputs*knChains, kzero), s_accum_dev(knInputs*knChains, kzero);


#ifdef __KISTI_GPU__
  logfile << "# Now we are in warming up..." << std::flush;
#else
  std::cout << "# Now we are in warming up..." << std::flush;
#endif


  smp_.warm_up(nwarmup);


#ifdef __KISTI_GPU__
  logfile << " done." << std::endl << std::flush;
  logfile << "# Measuring spin-spin correlation... (current/total)" << std::endl << std::flush;
#else
  std::cout << " done." << std::endl << std::flush;
  std::cout << "# Measuring spin-spin correlation... (current/total)" << std::endl << std::flush;
#endif


  const thrust::complex<FloatType> oneOverTotalMeas = 1/static_cast<FloatType>(nIterations*knChains);
  for (int n=0; n<nIterations; ++n)
  {


#ifdef __KISTI_GPU__
    logfile << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nIterations << std::flush;
#else
    std::cout << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nIterations << std::flush;
#endif


    smp_.do_mcmc_steps(nMCSteps);
    for (int i=0; i<knInputs; ++i)
    {
      // meas <\sigma^x_i>
      psi_.forward(i, PTR_FROM_THRUST(tmplnpsi_dev_.data()));
      gpu_kernel::meas__AccumPsi2OverPsi0__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(
        knChains, knInputs, smp_.get_lnpsi(),
        PTR_FROM_THRUST(tmplnpsi_dev_.data()),
        PTR_FROM_THRUST(&s_accum_dev[i*knChains]));

      // meas <\sigma^x_i \sigma^x_j>
      for (int j=(i+1); j<knInputs; ++j)
      {
        thrust::fill(spinPairFlipIdx_dev_.begin(), spinPairFlipIdx_dev_.end(), thrust::pair<int, int>(i, j));
        psi_.forward(spinPairFlipIdx_dev_, PTR_FROM_THRUST(tmplnpsi_dev_.data()));
        gpu_kernel::meas__AccumPsi2OverPsi0__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(
          knChains, knInputs, smp_.get_lnpsi(),
          PTR_FROM_THRUST(tmplnpsi_dev_.data()),
          PTR_FROM_THRUST(&ss_accum_dev[(i*knInputs+j)*knChains]));
      }
    }
  }


#ifdef __KISTI_GPU__
  logfile << std::endl;
  logfile.close();
#else
  std::cout << std::endl;
#endif


  cublas::gemm(theCublasHandle_, 1, knInputs*knInputs, knChains, oneOverTotalMeas, kzero,
    PTR_FROM_THRUST(kones.data()), PTR_FROM_THRUST(ss_accum_dev.data()), PTR_FROM_THRUST(ss_dev_.data()));
  cublas::gemm(theCublasHandle_, 1, knInputs, knChains, oneOverTotalMeas, kzero,
    PTR_FROM_THRUST(kones.data()), PTR_FROM_THRUST(s_accum_dev.data()), PTR_FROM_THRUST(s_dev_.data()));

  thrust::host_vector<thrust::complex<FloatType>> ss_host(ss_dev_), s_host(s_dev_);
  for (int i=0; i<knInputs; ++i)
  {
    s[i] = s_host[i].real();
    for (int j=0; j<knInputs; ++j)
    {
      const int idx = i*knInputs+j;
      ss[idx] = ss_host[idx].real();
    }
  }
}

namespace gpu_kernel
{
template <typename FloatType>
__global__ void meas__AccumPsi2OverPsi0__(
  const int nChains,
  const int nInputs,
  const thrust::complex<FloatType> * lnpsi_0,
  const thrust::complex<FloatType> * lnpsi_2,
  thrust::complex<FloatType> * ss)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned nstep = gridDim.x*blockDim.x;
  while (idx < nChains)
  {
    ss[idx] = ss[idx] + thrust::exp(lnpsi_2[idx]-lnpsi_0[idx]);
    idx += nstep;
  }
}
} // end namespace gpu_kernel


template <typename TraitsClass>
MeasSpontaneousMagnetization<TraitsClass>::MeasSpontaneousMagnetization(Sampler4SpinHalf<TraitsClass> & smp):
  smp_(smp),
  kzero(thrust::complex<FloatType>(0, 0)),
  kone(thrust::complex<FloatType>(1, 0)),
  koneOverNinputs(thrust::complex<FloatType>(1.0/smp_.get_nInputs(), 0)),
  kones(smp.get_nInputs(), thrust::complex<FloatType>(1, 0)),
  tmpmag_dev_(smp.get_nChains()),
  mag_dev_(smp.get_nChains())
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
}

template <typename TraitsClass>
MeasSpontaneousMagnetization<TraitsClass>::~MeasSpontaneousMagnetization()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename TraitsClass>
void MeasSpontaneousMagnetization<TraitsClass>::measure(const int nIterations, const int nMCSteps, const int nwarmup, FloatType & m1, FloatType & m2, FloatType & m4)
{
  std::cout << "# Now we are in warming up..." << std::flush;
  smp_.warm_up(nwarmup);
  std::cout << " done." << std::endl << std::flush;
  std::cout << "# Measuring spontaneous magnetization... (current/total)" << std::endl << std::flush;
  thrust::fill(tmpmag_dev_.begin(), tmpmag_dev_.end(), kzero);
  const FloatType oneOverTotalMeas = 1/static_cast<FloatType>(nIterations*smp_.get_nChains());
  m1 = kzero.real(), m2 = kzero.real(), m4 = kzero.real();
  for (int n=0; n<nIterations; ++n)
  {
    std::cout << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nIterations << std::flush;
    smp_.do_mcmc_steps(nMCSteps);
    cublas::gemm(theCublasHandle_, 1, smp_.get_nChains(), smp_.get_nInputs(), koneOverNinputs, kzero,
      PTR_FROM_THRUST(kones.data()), smp_.get_quantumStates(), PTR_FROM_THRUST(tmpmag_dev_.data()));
    // \sum_{i=1}(s_i) -> |\sum_{i=1}(s_i)|
    thrust::transform(tmpmag_dev_.begin(), tmpmag_dev_.end(), mag_dev_.begin(), internal_impl::ComplexABSFunctor<FloatType>());
    m1 += thrust::reduce(thrust::device, mag_dev_.begin(), mag_dev_.end(), kzero.real())*oneOverTotalMeas;
    m2 += thrust::inner_product(thrust::device, mag_dev_.begin(), mag_dev_.end(), mag_dev_.begin(), kzero.real())*oneOverTotalMeas;
    m4 += internal_impl::l4_norm(mag_dev_)*oneOverTotalMeas;
  }
  std::cout << std::endl;
}


template <typename TraitsClass>
MeasOrderParameter<TraitsClass>::MeasOrderParameter(Sampler4SpinHalf<TraitsClass> & smp,
  const thrust::host_vector<thrust::complex<FloatType>> coeff_host):
  smp_(smp),
  kzero(thrust::complex<FloatType>(0, 0)),
  kone(thrust::complex<FloatType>(1, 0)),
  koneOverNinputs(thrust::complex<FloatType>(1.0/smp_.get_nInputs(), 0)),
  coeff_dev_(smp.get_nInputs()),
  tmpmag_dev_(smp.get_nChains()),
  mag_dev_(smp.get_nChains())
{
  if (smp.get_nInputs() != coeff_host.size())
    throw std::invalid_argument("smp.get_nInputs() != coeff_host.size()");
  coeff_dev_ = coeff_host;
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
}

template <typename TraitsClass>
MeasOrderParameter<TraitsClass>::~MeasOrderParameter()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename TraitsClass>
void MeasOrderParameter<TraitsClass>::measure(const int nIterations, const int nMCSteps, const int nwarmup,
  FloatType & m1, FloatType & m2, FloatType & m4)
{
  std::cout << "# Now we are in warming up..." << std::flush;
  smp_.warm_up(nwarmup);
  std::cout << " done." << std::endl << std::flush;
  std::cout << "# Measuring spontaneous magnetization... (current/total)" << std::endl << std::flush;
  thrust::fill(tmpmag_dev_.begin(), tmpmag_dev_.end(), kzero);
  const FloatType oneOverTotalMeas = 1/static_cast<FloatType>(nIterations*smp_.get_nChains());
  m1 = kzero.real(), m2 = kzero.real(), m4 = kzero.real();
  for (int n=0; n<nIterations; ++n)
  {
    std::cout << "\r# --- " << std::setw(4) << (n+1) << " / " << std::setw(4) << nIterations << std::flush;
    smp_.do_mcmc_steps(nMCSteps);
    cublas::gemm(theCublasHandle_, 1, smp_.get_nChains(), smp_.get_nInputs(), koneOverNinputs, kzero,
      PTR_FROM_THRUST(coeff_dev_.data()), smp_.get_quantumStates(), PTR_FROM_THRUST(tmpmag_dev_.data()));
    // \sum_{i=1}(s_i) -> |\sum_{i=1}(s_i)|
    thrust::transform(tmpmag_dev_.begin(), tmpmag_dev_.end(), mag_dev_.begin(), internal_impl::ComplexABSFunctor<FloatType>());
    m1 += thrust::reduce(thrust::device, mag_dev_.begin(), mag_dev_.end(), kzero.real())*oneOverTotalMeas;
    m2 += thrust::inner_product(thrust::device, mag_dev_.begin(), mag_dev_.end(), mag_dev_.begin(), kzero.real())*oneOverTotalMeas;
    m4 += internal_impl::l4_norm(mag_dev_)*oneOverTotalMeas;
  }
  std::cout << std::endl;
}


namespace fermion
{
namespace jordanwigner
{
template <typename TraitsClass>
Sampler4SpinHalf<TraitsClass>::Sampler4SpinHalf(AnsatzType & psi, const std::array<int, 2> & np,
  const unsigned long seedNumber, const unsigned long seedDistance):
  BaseParallelSampler<fermion::jordanwigner::Sampler4SpinHalf, TraitsClass>(psi.get_nInputs(), psi.get_nChains(), seedNumber, seedDistance),
  psi_(psi),
  np_(np),
  knSites(psi.get_nInputs()/2),
  exchanger_(psi.get_nChains(), psi.get_nInputs(), seedNumber*12345ul, seedDistance),
  spinPairIdx_dev_(psi.get_nChains())
{
  // ranges of machine inputs:
  // [0~knSites) -> spin up; [knSites~2*knSites) -> spin down
  if (psi_.get_nInputs()%2 != 0)
    throw std::invalid_argument("psi_.get_nInputs()%2 != 0");
}

template <typename TraitsClass>
void Sampler4SpinHalf<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  // +1 : particle is filling at the site.
  // -1 : particle is empty at the site.
  thrust::host_vector<thrust::complex<FloatType> > spinStates_host(psi_.get_nChains()*psi_.get_nInputs(), thrust::complex<FloatType>(-1.0, 0.0));
  std::vector<int> idx(knSites);
  for (int i=0; i<idx.size(); ++i)
    idx[i] = i;
  for (int k=0; k<psi_.get_nChains(); ++k)
  {
    // s : 0 (spin up), 1 (spin down)
    for (int s=0; s<2; ++s)
    {
      std::shuffle(idx.begin(), idx.end(), std::default_random_engine((12345u*k+9876543210u*s)));
      for (int n=0; n<np_[s]; ++n)
        spinStates_host[k*psi_.get_nInputs()+s*knSites+idx[n]] = thrust::complex<FloatType>(1.0, 0.0);
    }
  }
  thrust::device_vector<thrust::complex<FloatType> > spinStates_dev(spinStates_host);
  psi_.initialize(lnpsi_dev, PTR_FROM_THRUST(spinStates_dev.data()));
  // initialize spin-exchange sampler
  exchanger_.init(kawasaki::IsBondState(), spinStates_dev.data());
  exchanger_.get_indexes_of_spin_pairs(spinPairIdx_dev_);
}

template <typename TraitsClass>
void Sampler4SpinHalf<TraitsClass>::sampling_(thrust::complex<FloatType> * lnpsi_dev)
{
  exchanger_.get_indexes_of_spin_pairs(spinPairIdx_dev_);
  psi_.forward(spinPairIdx_dev_, lnpsi_dev);
}

template <typename TraitsClass>
void Sampler4SpinHalf<TraitsClass>::accept_next_state_(bool * isNewStateAccepted_dev)
{
  psi_.spin_flip(isNewStateAccepted_dev, spinPairIdx_dev_);
  exchanger_.do_exchange(isNewStateAccepted_dev);
}


template <typename TraitsClass>
MeasOPDM<TraitsClass>::MeasOPDM(fermion::jordanwigner::Sampler4SpinHalf<TraitsClass> & smp, AnsatzType & psi):
  smp_(smp),
  psi_(psi),
  tmpspinStates_dev_(psi.get_nChains()*psi.get_nInputs()),
  tmplnpsi_dev_(psi.get_nChains()),
  OPDM_dev_(psi.get_nChains()),
  knChains(psi.get_nChains()),
  knSites(psi.get_nInputs()/2),
  kgpuBlockSize(CHECK_BLOCK_SIZE(1+(smp.get_nChains()-1)/NUM_THREADS_PER_BLOCK))
{
  if (smp.get_nInputs() != psi.get_nInputs())
    throw std::invalid_argument("smp.get_nInputs() != psi.get_nInputs()");
  if (smp.get_nChains() != psi.get_nChains())
    throw std::invalid_argument("smp.get_nChains() != psi.get_nChains()");
}

// OPDM = <\psi|c^+_{n+m,up}c^+_{n+m,down}c_{n,down}c_{n,up}|\psi>
template <typename TraitsClass>
std::complex<typename TraitsClass::FloatType> MeasOPDM<TraitsClass>::measure(const int n, const int m,
  const int nIterations, const int nMCSteps, const int nwarmup)
{
  if ((n+m) >= knSites)
    throw std::invalid_argument("(n+m) >= knSites");
  std::cout << "# OPDM = <\\psi|c^+_{" << n+m << ",up}c^+_{" << n+m << ",down}c_{"
    << n << ",down}c_{" << n << ",up}|\\psi>" << std::endl;
  std::cout << "# Now we are in warming up..." << std::flush;
  smp_.warm_up(nwarmup);
  std::cout << " done." << std::endl << std::flush;
  std::cout << "# Measuring OPDM ... (current/total)" << std::endl << std::flush;
  thrust::complex<FloatType> SumOfOPDM(0.0);
  for (int niter=0; niter<nIterations; ++niter)
  {
    std::cout << "\r# --- " << std::setw(4) << (niter+1) << " / " << std::setw(4) << nIterations << std::flush;
    smp_.do_mcmc_steps(nMCSteps);
    // OPDM = 1/16*(1+s_{n+m})*(1+s_{n+m+nSites})*(1-s_{n})*(1-s_{n+nSites})*\prod_{l=n}^{n+m-1}(s_{l}*s_{l+nSites})*psi1/psi0
    if (m != 0)
    {
      CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(tmpspinStates_dev_.data()),
        smp_.get_quantumStates(), sizeof(thrust::complex<FloatType>)*tmpspinStates_dev_.size(), cudaMemcpyDeviceToDevice));
      // flip s_{n+m},s_{n+m+nSites},s_{n},and s_{n+nSites}
      gpu_kernel::OPDM__FlipSpins__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, knSites,
        n, m, PTR_FROM_THRUST(tmpspinStates_dev_.data()));
      // lnpsi_1 = <...-s_{n}...-s_{n+m}...-s_{n+nSites}...-s_{n+m+nSites}|\psi>
      psi_.forward(PTR_FROM_THRUST(tmpspinStates_dev_.data()), PTR_FROM_THRUST(tmplnpsi_dev_.data()), false);
      gpu_kernel::meas__OPDM__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, knSites, n, m,
        smp_.get_quantumStates(), smp_.get_lnpsi(), PTR_FROM_THRUST(tmplnpsi_dev_.data()), PTR_FROM_THRUST(OPDM_dev_.data()));
    }
    else
      gpu_kernel::meas__OPDM__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, knSites, n,
        smp_.get_quantumStates(), PTR_FROM_THRUST(OPDM_dev_.data()));
    SumOfOPDM = thrust::reduce(thrust::device, OPDM_dev_.begin(), OPDM_dev_.end(), SumOfOPDM);
  }
  std::cout << std::endl;
  SumOfOPDM /= static_cast<FloatType>((nIterations*psi_.get_nChains()));
  return std::complex<FloatType>(SumOfOPDM.real(), SumOfOPDM.imag());
}
} // end namespace jordanwigner
} // end namespace fermion

namespace gpu_kernel
{
template <typename FloatType>
__global__ void OPDM__FlipSpins__(const int nChains, const int nSites, const int n, const int m, thrust::complex<FloatType> * spinStates)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned nstep = gridDim.x*blockDim.x;
  while (idx < nChains)
  {
    const int k = idx;
    for (int s=0; s<2; ++s)
    {
      const int idx = k*2*nSites+s*nSites+n+m;
      spinStates[idx] = -spinStates[idx].real();
      spinStates[idx-m] = -spinStates[idx-m].real();
    }
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void meas__OPDM__(const int nChains, const int nSites, const int n, const int m,
  const thrust::complex<FloatType> * spinStates, const thrust::complex<FloatType> * lnpsi0,
  const thrust::complex<FloatType> * lnpsi1, thrust::complex<FloatType> * OPDM)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned nstep = gridDim.x*blockDim.x;
  const FloatType zero = 0, one = 1, fac = 1.0/16.0;
  while (idx < nChains)
  {
    const int k = idx, kup = k*2*nSites, kdw = k*2*nSites+nSites;
    OPDM[k] = zero;
    FloatType spinProduct = one;
    for (int l=n+1; l<n+m; ++l)
      spinProduct *= (spinStates[kup+l].real()*spinStates[kdw+l].real());
    OPDM[k] = fac*(one+spinStates[kup+n+m].real())*(one+spinStates[kdw+n+m].real())*
                  (one-spinStates[kup+n].real())*(one-spinStates[kdw+n].real())*
                  spinProduct*thrust::exp(lnpsi1[k]-lnpsi0[k]);
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void meas__OPDM__(const int nChains, const int nSites, const int n,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * OPDM)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned nstep = gridDim.x*blockDim.x;
  const FloatType one = 1.0, fac = 0.25;
  while (idx < nChains)
  {
    const int k = idx, kup = k*2*nSites, kdw = k*2*nSites+nSites;
    OPDM[k] = fac*(one+spinStates[kup+n].real())*(one+spinStates[kdw+n].real());
    idx += nstep;
  }
}
} // end namespace gpu_kernel
