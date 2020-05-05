// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include "common.cuh"
#include "trng4cuda.cuh"

/*
 * Base class of importance sampling for wave functions: ln(\psi(x))
 *  - ratio = norm(ln(\psi(x1))-ln(\psi(x0)))
 *   where x1 is a candidate of the next state and x0 is a current state.
 */
template <template<typename> class DerivedParallelSampler, typename TraitsClass>
class BaseParallelSampler
{
  using FloatType = typename TraitsClass::FloatType;
public:
  BaseParallelSampler(const uint32_t nMCUnitSteps, const uint32_t nChains, const uint64_t seedNumber, const uint64_t seedDistance);
  void warm_up(const uint32_t nMCSteps = 100u);
  void do_mcmc_steps(const uint32_t nMCSteps = 1u);
  void get_htilda(thrust::complex<FloatType> * htilda_dev);
  void get_lnpsiGradients(thrust::complex<FloatType> * lnpsiGradients_dev);
  uint32_t get_nChains() const { return knChains; }
  void evolve(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate);
  void save() const;
private:
  const uint32_t knMCUnitSteps, knChains, kgpuBlockSize;
  thrust::device_vector<bool> isNewStateAccepted_dev_;
  thrust::device_vector<FloatType> rngValues_dev_;
  TRNGWrapper<FloatType, trng::yarn2> rng_;
protected:
  thrust::device_vector<thrust::complex<FloatType>> lnpsi1_dev_, lnpsi0_dev_;
};

namespace gpu_kernel
{
template <typename FloatType>
__global__ void Sampler__ParallelMetropolisUpdate__(
  const FloatType * rngValues,
  const uint32_t nChains,
  const thrust::complex<FloatType> * lnpsi1,
  thrust::complex<FloatType> * lnpsi0,
  bool * isNewStateAccepted
);
}


/*
 * return a randomly shuffled array: [0,1,2,3,4,5,...] => [[4,0,1,...],[2,3,5,...],[...],...]
 */
class RandomBatchIndexing
{
public:
  RandomBatchIndexing(const uint32_t size, const double rate, unsigned seed = 0u):
    fullBatch_(size),
    miniBatches_(size/static_cast<uint32_t>(size*rate)+(size%static_cast<uint32_t>(size*rate)!=0)),
    miniBatchesIdx_(0u),
    rng_(seed)
  {
    if (rate <= 0 || rate > 1)
      throw std::invalid_argument("rate <= 0 or rate > 1");
    const uint32_t pSize = static_cast<uint32_t>(size*rate);
    if (pSize == 0)
      throw std::invalid_argument("(size*rate)<1");
    for (uint32_t j=0u; j<fullBatch_.size(); ++j)
      fullBatch_[j] = j;
    std::shuffle(fullBatch_.begin(), fullBatch_.end(), rng_);
    for (uint32_t i=0u; i<miniBatches_.size()-1; ++i)
      miniBatches_[i].assign(pSize, 0);
    if (size%pSize != 0)
      miniBatches_[miniBatches_.size()-1].assign(size%pSize, 0);
    else
      miniBatches_[miniBatches_.size()-1].assign(pSize, 0);
    uint32_t fullBatchIdx = 0u;
    for (auto & miniBatch : miniBatches_)
    {
      for (auto & index : miniBatch)
        index = fullBatch_[fullBatchIdx++];
      std::sort(miniBatch.begin(), miniBatch.end());
    }
  }

  const thrust::host_vector<uint32_t> & get_miniBatch() const
  {
    return miniBatches_[miniBatchesIdx_];
  }

  void next()
  {
    miniBatchesIdx_ += 1u;
    if (miniBatchesIdx_ == miniBatches_.size())
    {
      miniBatchesIdx_ = 0u;
      std::shuffle(fullBatch_.begin(), fullBatch_.end(), rng_);
      uint32_t fullBatchIdx = 0u;
      for (auto & miniBatch : miniBatches_)
      {
        for (auto & index : miniBatch)
          index = fullBatch_[fullBatchIdx++];
        std::sort(miniBatch.begin(), miniBatch.end());
      }
    }
  }

private:
  std::vector<uint32_t> fullBatch_;
  std::vector<thrust::host_vector<uint32_t>> miniBatches_;
  uint32_t miniBatchesIdx_;
  std::mt19937 rng_;
};

#include "impl_mcmc_sampler.cuh"
