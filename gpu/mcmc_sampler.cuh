// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <vector>
#include <random>
#include <trng/yarn2.hpp>
#include <trng/yarn5.hpp>
#include <trng/yarn5s.hpp>
#include <trng/uniform01_dist.hpp>
#include <trng/uniform_int_dist.hpp>
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
  CurandWrapper<FloatType, CURAND_RNG_PSEUDO_MTGP32> rng_;
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
template <typename RandEngineType>
class RandomBatchIndexing
{
public:
  RandomBatchIndexing(const int size, const double rate);
  const thrust::host_vector<int> & get_miniBatch() const;
  void next();
private:
  std::vector<int> indexOfFullBatch_;
  std::vector<thrust::host_vector<int>> indexOfMiniBatch_;
  int batchSetIdx_;
  RandEngineType rng_;
};

#include "impl_mcmc_sampler.cuh"
