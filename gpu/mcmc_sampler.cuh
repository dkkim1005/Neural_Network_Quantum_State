// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include "common.cuh"
#include "curand_wrapper.cuh"

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
  BaseParallelSampler(const int nMCUnitSteps, const int nChains, const unsigned long long seedNumber = 0ull);
  void warm_up(const int nMCSteps = 100);
  void do_mcmc_steps(const int nMCSteps = 1);
  void get_htilda(thrust::complex<FloatType> * htilda_dev);
  void get_lnpsiGradients(thrust::complex<FloatType> * lnpsiGradients_dev);
  int get_nChains() const { return knChains; }
  void evolve(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate);
private:
  const int knMCUnitSteps, knChains, kgpuBlockSize;
  thrust::device_vector<bool> isNewStateAccepted_dev_;
  thrust::device_vector<FloatType> rngValues_dev_;
  CurandWrapper<FloatType, CURAND_RNG_PSEUDO_PHILOX4_32_10> rng_;
protected:
  thrust::device_vector<thrust::complex<FloatType>> lnpsi1_dev_, lnpsi0_dev_;
};

namespace gpu_kernel
{
template <typename FloatType>
__global__ void Sampler__ParallelMetropolisUpdate__(
  const FloatType * rngValues,
  const int nChains,
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
  RandomBatchIndexing(const int size, const double rate, unsigned seed = 0u):
    indexOfFullBatch_(size),
    indexOfMiniBatch_(size/static_cast<int>(size*rate)+(size%static_cast<int>(size*rate)!=0)),
    batchSetIdx_(0),
    rng_(seed)
  {
    if (rate <= 0 || rate > 1)
      throw std::invalid_argument("rate <= 0 or rate > 1");
    const int partialSize = static_cast<int>(size*rate);
    if (partialSize == 0)
      throw std::invalid_argument("(size*rate)<1");
    for (int j=0; j<indexOfFullBatch_.size(); ++j)
      indexOfFullBatch_[j] = j;
    std::shuffle(indexOfFullBatch_.begin(), indexOfFullBatch_.end(), rng_);
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

  const thrust::host_vector<int> & get_miniBatch() const
  {
    return indexOfMiniBatch_[batchSetIdx_];
  }

  void next()
  {
    batchSetIdx_ += 1;
    if (batchSetIdx_ == indexOfMiniBatch_.size())
    {
      batchSetIdx_ = 0;
      std::shuffle(indexOfFullBatch_.begin(), indexOfFullBatch_.end(), rng_);
      int nodeIdx = 0;
      for (auto & index : indexOfMiniBatch_)
        for (auto & item : index)
          item = indexOfFullBatch_[nodeIdx++];
    }
  }

private:
  std::vector<int> indexOfFullBatch_;
  std::vector<thrust::host_vector<int>> indexOfMiniBatch_;
  int batchSetIdx_;
  std::mt19937 rng_;
};

#include "impl_mcmc_sampler.cuh"
