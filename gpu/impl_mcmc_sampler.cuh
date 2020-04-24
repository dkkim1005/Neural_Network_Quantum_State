// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
BaseParallelSampler<DerivedParallelSampler, TraitsClass>::BaseParallelSampler(const int nMCUnitSteps,
  const int nChains, const unsigned long long seedNumber):
  knMCUnitSteps(nMCUnitSteps),
  knChains(nChains),
  isNewStateAccepted_dev_(nChains, true),
  lnpsi0_dev_(nChains),
  lnpsi1_dev_(nChains),
  rngValues_dev_(nChains),
  rng_(seedNumber),
  kgpuBlockSize(1+(nChains-1)/NUM_THREADS_PER_BLOCK) {}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::warm_up(const int nMCSteps)
{
  // prepare an initial state
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->initialize(PTR_FROM_THRUST(lnpsi0_dev_.data()));
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->accept_next_state(PTR_FROM_THRUST(isNewStateAccepted_dev_.data()));
  // MCMC sampling for warming up
  this->do_mcmc_steps(nMCSteps);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::do_mcmc_steps(const int nMCSteps)
{
  // Markov chain MonteCarlo(MCMC) sampling with nskip iterations
  for (int n=0; n<(nMCSteps*knMCUnitSteps); ++n)
  {
    static_cast<DerivedParallelSampler<TraitsClass>*>(this)->sampling(PTR_FROM_THRUST(lnpsi1_dev_.data()));
    rng_.get_uniformDist(PTR_FROM_THRUST(rngValues_dev_.data()), knChains);
    gpu_kernel::Sampler__ParallelMetropolisUpdate__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(rngValues_dev_.data()), knChains,
      PTR_FROM_THRUST(lnpsi1_dev_.data()), PTR_FROM_THRUST(lnpsi0_dev_.data()), PTR_FROM_THRUST(isNewStateAccepted_dev_.data()));
    static_cast<DerivedParallelSampler<TraitsClass>*>(this)->accept_next_state(PTR_FROM_THRUST(isNewStateAccepted_dev_.data()));
  }
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::get_htilda(thrust::complex<FloatType> * htilda_dev)
{
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->get_htilda(PTR_FROM_THRUST(lnpsi0_dev_.data()),
    PTR_FROM_THRUST(lnpsi1_dev_.data()), htilda_dev);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::get_lnpsiGradients(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->get_lnpsiGradients(lnpsiGradients_dev);
}

template <template<typename> class DerivedParallelSampler, typename TraitsClass>
void BaseParallelSampler<DerivedParallelSampler, TraitsClass>::evolve(const thrust::complex<FloatType> * trueGradients_dev,
  const FloatType learningRate)
{
  static_cast<DerivedParallelSampler<TraitsClass>*>(this)->evolve(trueGradients_dev, learningRate);
}

namespace gpu_kernel
{
template <typename FloatType>
__global__ void Sampler__ParallelMetropolisUpdate__(
  const FloatType * rngValues,
  const int nChains,
  const thrust::complex<FloatType> * lnpsi1,
  thrust::complex<FloatType> * lnpsi0,
  bool * isNewStateAccepted)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    const FloatType ratio = thrust::norm(thrust::exp(lnpsi1[idx]-lnpsi0[idx]));
    isNewStateAccepted[idx] = (rngValues[idx]<ratio);
    lnpsi0[idx] = (isNewStateAccepted[idx] ? lnpsi1[idx] : lnpsi0[idx]);
    idx += nstep;
  }
}
}


template <typename RandEngineType>
RandomBatchIndexing<RandEngineType>::RandomBatchIndexing(const int size, const double rate):
  indexOfFullBatch_(size),
  indexOfMiniBatch_(size/static_cast<int>(size*rate)+(size%static_cast<int>(size*rate)!=0)),
  batchSetIdx_(0)
{
  if (rate <= 0 || rate > 1)
    throw std::invalid_argument("rate <= 0 or rate > 1");
  const int partialSize = static_cast<int>(size*rate);
  if (partialSize == 0)
    throw std::invalid_argument("(size*rate)<1");
  for (int j=0; j<indexOfFullBatch_.size(); ++j)
    indexOfFullBatch_[j] = j;
  std::random_shuffle(indexOfFullBatch_.begin(), indexOfFullBatch_.end(), rng_);
  for (int i=0; i<indexOfMiniBatch_.size()-1; ++i)
    indexOfMiniBatch_[i].assign(partialSize, 0);
  if (size%partialSize != 0)
    indexOfMiniBatch_[indexOfMiniBatch_.size()-1].assign(size%partialSize, 0);
  else
    indexOfMiniBatch_[indexOfMiniBatch_.size()-1].assign(partialSize, 0);
  int nodeIdx = 0;
  for (auto & index : indexOfMiniBatch_)
    for (auto & item : index)
      item = indexOfFullBatch_[nodeIdx++];
}

template <typename RandEngineType>
const thrust::host_vector<int> & RandomBatchIndexing<RandEngineType>::get_miniBatch() const
{
  return indexOfMiniBatch_[batchSetIdx_];
}

template <typename RandEngineType>
void RandomBatchIndexing<RandEngineType>::next()
{
  batchSetIdx_ += 1;
  if (batchSetIdx_ == indexOfMiniBatch_.size())
  {
    batchSetIdx_ = 0;
    std::random_shuffle(indexOfFullBatch_.begin(), indexOfFullBatch_.end(), rng_);
    int nodeIdx = 0;
    for (auto & index : indexOfMiniBatch_)
      for (auto & item : index)
        item = indexOfFullBatch_[nodeIdx++];
  }
}
