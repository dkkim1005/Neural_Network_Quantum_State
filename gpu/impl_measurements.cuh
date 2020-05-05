#pragma once

namespace spinhalf
{
template <typename TraitsClass>
MeasMagnetizationZ<TraitsClass>::MeasMagnetizationZ(AnsatzType & machine, const uint64_t seedDistance, const uint64_t seedNumber):
  BaseParallelSampler<MeasMagnetizationZ, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedNumber, seedDistance),
  idx_(0u),
  machine_(machine),
  knInputs(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  m1_(machine.get_nChains(), 0),
  m2_(machine.get_nChains(), 0),
  m4_(machine.get_nChains(), 0),
  kgpuBlockSize(CHECK_BLOCK_SIZE(1u+(machine.get_nChains()-1u)/NUM_THREADS_PER_BLOCK)),
  kzero(0) {}

template <typename TraitsClass>
void MeasMagnetizationZ<TraitsClass>::meas(const uint32_t nTrials, const uint32_t nwarms, const uint32_t nMCSteps, magnetization<FloatType> & outputs)
{
  std::cout << "# Now we are in warming up...(" << nwarms << ")" << std::endl << std::flush;
  this->warm_up(nwarms);
  std::cout << "# # of total measurements:" << nTrials*knChains << std::endl << std::flush;
  std::vector<FloatType> m1arr(nTrials, kzero), m2arr(nTrials, kzero), m4arr(nTrials, kzero), mtemp(knChains, kzero);
  const FloatType invNinputs = 1/static_cast<FloatType>(knInputs);
  const FloatType invNchains = 1/static_cast<FloatType>(knChains);
  const FloatType invNtrials = 1/static_cast<FloatType>(nTrials);
  for (uint32_t n=0u; n<nTrials; ++n)
  {
    std::cout << "# " << (n+1) << " / " << nTrials << std::endl << std::flush;
    this->do_mcmc_steps(nMCSteps);
    const thrust::complex<FloatType> * spinStates_dev = machine_.get_spinStates();
    gpu_kernel::meas__MeasAbsMagZ__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(spinStates_dev, knInputs, knChains,
      PTR_FROM_THRUST(m1_.data()), PTR_FROM_THRUST(m2_.data()), PTR_FROM_THRUST(m4_.data()));
    m1arr[n] = thrust::reduce(thrust::device, m1_.begin(), m1_.end(), kzero)*invNinputs*invNchains;
    m2arr[n] = thrust::reduce(thrust::device, m2_.begin(), m2_.end(), kzero)*std::pow(invNinputs, 2)*invNchains;
    m4arr[n] = thrust::reduce(thrust::device, m4_.begin(), m4_.end(), kzero)*std::pow(invNinputs, 4)*invNchains;
  }
  outputs.m1 = std::accumulate(m1arr.begin(), m1arr.end(), kzero)*invNtrials;
  outputs.m2 = std::accumulate(m2arr.begin(), m2arr.end(), kzero)*invNtrials;
  outputs.m4 = std::accumulate(m4arr.begin(), m4arr.end(), kzero)*invNtrials;
}

template <typename TraitsClass>
void MeasMagnetizationZ<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.initialize(lnpsi_dev);
}

template <typename TraitsClass>
void MeasMagnetizationZ<TraitsClass>::sampling_(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.forward(idx_++, lnpsi_dev);
  idx_ = ((idx_ == knInputs) ? 0u : idx_);
}

template <typename TraitsClass>
void MeasMagnetizationZ<TraitsClass>::accept_next_state_(const bool * isNewStateAccepted_dev)
{
  machine_.spin_flip(isNewStateAccepted_dev);
}


template <typename TraitsClass>
MeasMagnetizationX<TraitsClass>::MeasMagnetizationX(AnsatzType & machine, const uint64_t seedDistance, const uint64_t seedNumber):
  BaseParallelSampler<MeasMagnetizationX, TraitsClass>(machine.get_nInputs(), machine.get_nChains(), seedNumber, seedDistance),
  machine_(machine),
  idx_(0u),
  knInputs(machine.get_nInputs()),
  knChains(machine.get_nChains()),
  kgpuBlockSize(CHECK_BLOCK_SIZE(1u+(machine.get_nChains()-1u)/NUM_THREADS_PER_BLOCK)),
  kzero(0) {}

template <typename TraitsClass>
void MeasMagnetizationX<TraitsClass>::meas(const uint32_t nTrials, const uint32_t nwarms, const uint32_t nMCSteps, magnetization<FloatType> & outputs)
{
  std::cout << "# Now we are in warming up...(" << nwarms << ")" << std::endl << std::flush;
  this->warm_up(nwarms);
  std::cout << "# # of total measurements:" << nTrials*knChains << std::endl << std::flush;
  FloatType mx1 = kzero, mx2 = kzero;
  thrust::device_vector<FloatType> mx1temp_dev(knChains, kzero), mx2temp_dev(knChains, kzero);
  const FloatType invNinputs = 1/static_cast<FloatType>(knInputs);
  const FloatType invNchains = 1/static_cast<FloatType>(knChains);
  const FloatType invNtrials = 1/static_cast<FloatType>(nTrials);
  const thrust::device_vector<bool> IsSpinFliped_dev(knChains, true);
  for (uint32_t n=0u; n<nTrials; ++n)
  {
    std::cout << "# " << (n+1) << " / " << nTrials << std::endl << std::flush;
    this->do_mcmc_steps(nMCSteps);
    thrust::fill(mx1temp_dev.begin(), mx1temp_dev.end(), kzero);
    thrust::fill(mx2temp_dev.begin(), mx2temp_dev.end(), kzero);
    for (uint32_t i=0u; i<knInputs; ++i)
    {
      machine_.forward(i, PTR_FROM_THRUST(lnpsi1_dev_.data()));
      gpu_kernel::meas__AccPsi1OverPsi0__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(lnpsi0_dev_.data()),
        PTR_FROM_THRUST(lnpsi1_dev_.data()), knChains, PTR_FROM_THRUST(mx1temp_dev.data()));
    }
    for (uint32_t i=0u; i<knInputs; ++i)
    {
      machine_.spin_flip(PTR_FROM_THRUST(IsSpinFliped_dev.data()), i);
      for (uint32_t j=0u; j<knInputs; ++j)
      {
        machine_.forward(j, PTR_FROM_THRUST(lnpsi1_dev_.data()));
        gpu_kernel::meas__AccPsi1OverPsi0__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(lnpsi0_dev_.data()),
          PTR_FROM_THRUST(lnpsi1_dev_.data()), knChains, PTR_FROM_THRUST(mx2temp_dev.data()));
      }
      machine_.spin_flip(PTR_FROM_THRUST(IsSpinFliped_dev.data()), i);
    }
    mx1 += thrust::reduce(thrust::device, mx1temp_dev.begin(), mx1temp_dev.end(), kzero);
    mx2 += thrust::reduce(thrust::device, mx2temp_dev.begin(), mx2temp_dev.end(), kzero);
  }
  mx1 *= (invNinputs*invNchains*invNtrials);
  mx2 *= (std::pow(invNinputs, 2)*invNchains*invNtrials);
  outputs.m1 = mx1;
  outputs.m2 = mx2;
}

template <typename TraitsClass>
void MeasMagnetizationX<TraitsClass>::initialize_(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.initialize(lnpsi_dev);
}

template <typename TraitsClass>
void MeasMagnetizationX<TraitsClass>::sampling_(thrust::complex<FloatType> * lnpsi_dev)
{
  machine_.forward(idx_++, lnpsi_dev);
  idx_ = ((idx_ == knInputs) ? 0u : idx_);
}

template <typename TraitsClass>
void MeasMagnetizationX<TraitsClass>::accept_next_state_(const bool * isNewStateAccepted_dev)
{
  machine_.spin_flip(isNewStateAccepted_dev);
}
} // spinhalf


namespace gpu_kernel
{
template <typename FloatType>
__global__ void meas__MeasAbsMagZ__(
  const thrust::complex<FloatType> * spinStates,
  const uint32_t nInputs,
  const uint32_t nChains,
  FloatType * m1,
  FloatType * m2,
  FloatType * m4)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    m1[idx] = 0;
    for (uint32_t i=0u; i<nInputs; ++i)
      m1[idx] += spinStates[idx*nInputs+i].real();
    m1[idx] = std::abs(m1[idx]);
    m2[idx] = m1[idx]*m1[idx];
    m4[idx] = m2[idx]*m2[idx];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void meas__AccPsi1OverPsi0__(
  const thrust::complex<FloatType> * lnpsi0,
  const thrust::complex<FloatType> * lnpsi1,
  const uint32_t nChains,
  FloatType * mx)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    mx[idx] += (thrust::exp(lnpsi1[idx]-lnpsi0[idx])).real();
    idx += nstep;
  }
}
} // gpu_kernel
