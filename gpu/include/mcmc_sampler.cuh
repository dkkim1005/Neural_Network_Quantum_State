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
  BaseParallelSampler(const int nMCUnitSteps, const int nChains, const unsigned long seedNumber, const unsigned long seedDistance);
  void warm_up(const int nMCSteps = 100);
  void do_mcmc_steps(const int nMCSteps = 1);
  const thrust::complex<FloatType> * get_lnpsi() { return PTR_FROM_THRUST(lnpsi0_dev_.data()); }
  void get_htilda(thrust::complex<FloatType> * htilda_dev);
  void get_lnpsiGradients(thrust::complex<FloatType> * lnpsiGradients_dev);
  int get_nChains() const { return knChains; }
  void evolve(const thrust::complex<FloatType> * trueGradients_dev, const FloatType learningRate);
  void save() const;
private:
  const int knMCUnitSteps, knChains, kgpuBlockSize;
  thrust::device_vector<bool> isNewStateAccepted_dev_;
  thrust::device_vector<FloatType> rngValues_dev_;
  TRNGWrapper<FloatType, trng::yarn2> rng_;
protected:
  thrust::device_vector<thrust::complex<FloatType>> lnpsi1_dev_, lnpsi0_dev_;
};

namespace gpu_device
{
__device__ float Psi1OverPsi0Squared(const float Relnpsi0, const float Relnpsi1);
__device__ double Psi1OverPsi0Squared(const double Relnpsi0, const double Relnpsi1);
}

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
/*
class RandomBatchIndexing
{
public:
  RandomBatchIndexing(const int size, const double rate, unsigned seed = 0u):
    fullBatch_(size),
    miniBatches_(size/static_cast<int>(size*rate)+(size%static_cast<int>(size*rate)!=0)),
    miniBatchesIdx_(0),
    rng_(seed)
  {
    if (rate <= 0 || rate > 1)
      throw std::invalid_argument("rate <= 0 or rate > 1");
    const int pSize = static_cast<int>(size*rate);
    if (pSize == 0)
      throw std::invalid_argument("(size*rate)<1");
    for (int j=0; j<fullBatch_.size(); ++j)
      fullBatch_[j] = j;
    std::shuffle(fullBatch_.begin(), fullBatch_.end(), rng_);
    for (int i=0; i<miniBatches_.size()-1; ++i)
      miniBatches_[i].assign(pSize, 0);
    if (size%pSize != 0)
      miniBatches_[miniBatches_.size()-1].assign(size%pSize, 0);
    else
      miniBatches_[miniBatches_.size()-1].assign(pSize, 0);
    int fullBatchIdx = 0;
    for (auto & miniBatch : miniBatches_)
    {
      for (auto & index : miniBatch)
        index = fullBatch_[fullBatchIdx++];
      std::sort(miniBatch.begin(), miniBatch.end());
    }
  }

  const thrust::host_vector<int> & get_miniBatch() const
  {
    return miniBatches_[miniBatchesIdx_];
  }

  void next()
  {
    miniBatchesIdx_ += 1;
    if (miniBatchesIdx_ == miniBatches_.size())
    {
      miniBatchesIdx_ = 0;
      std::shuffle(fullBatch_.begin(), fullBatch_.end(), rng_);
      int fullBatchIdx = 0;
      for (auto & miniBatch : miniBatches_)
      {
        for (auto & index : miniBatch)
          index = fullBatch_[fullBatchIdx++];
        std::sort(miniBatch.begin(), miniBatch.end());
      }
    }
  }

private:
  std::vector<int> fullBatch_;
  std::vector<thrust::host_vector<int>> miniBatches_;
  int miniBatchesIdx_;
  std::mt19937 rng_;
};
*/
#include "impl_mcmc_sampler.cuh"
