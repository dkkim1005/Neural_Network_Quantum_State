// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <vector>
#include <trng/yarn2.hpp>
#include <trng/yarn5.hpp>
#include <trng/yarn5s.hpp>
#include <trng/uniform01_dist.hpp>
#include "common.cuh"

namespace gpu_kernel
{
template <typename FloatType, typename RNGType>
__global__ void rand__GenerateUniformDist__(const int size, RNGType * rng, FloatType * rngValues)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  trng::uniform01_dist<FloatType> uniformDist;
  while (idx < size)
  {
    rngValues[idx] = uniformDist(rng[idx]);
    idx += nstep;
  }
}
}

template <typename FloatType, typename RNGType>
class TRNGWrapper
{
public:
  TRNGWrapper(const unsigned long seedNuber, const unsigned seedDistance, const int nChains);
  ~TRNGWrapper();
  void get_uniformDist(FloatType * rngValues_dev);
private:
  RNGType * rng_dev_;
  const int knChains, kgpuBlockSize;
};

template <typename FloatType, typename RNGType>
TRNGWrapper<FloatType, RNGType>::TRNGWrapper(const unsigned long seedNuber, const unsigned seedDistance, const int nChains):
  knChains(nChains),
  kgpuBlockSize(CHECK_BLOCK_SIZE(1+(nChains-1)/NUM_THREADS_PER_BLOCK))
{
  std::vector<RNGType> rng(nChains);
  CHECK_ERROR(cudaSuccess, cudaMalloc(&rng_dev_, sizeof(RNGType)*nChains));
  for (int k=0; k<nChains; ++k)
  {
    rng[k].seed(seedNuber);
    rng[k].jump(2*seedDistance*k);
  }
  CHECK_ERROR(cudaSuccess, cudaMemcpy(rng_dev_, rng.data(), sizeof(RNGType)*nChains, cudaMemcpyHostToDevice));
}

template <typename FloatType, typename RNGType>
TRNGWrapper<FloatType, RNGType>::~TRNGWrapper()
{
  CHECK_ERROR(cudaSuccess, cudaFree(rng_dev_));
}

template <typename FloatType, typename RNGType>
void TRNGWrapper<FloatType, RNGType>::get_uniformDist(FloatType * rngValues_dev)
{
  gpu_kernel::rand__GenerateUniformDist__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, rng_dev_, rngValues_dev);
}
