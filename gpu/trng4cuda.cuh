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
__global__ void rand__GenerateUniformDist__(const uint32_t size, RNGType * rng, FloatType * rngValues)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
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
  TRNGWrapper(const uint64_t seedNuber, const uint64_t seedDistance, const uint32_t nChains);
  ~TRNGWrapper();
  void get_uniformDist(FloatType * rngValues_dev);
private:
  RNGType * rng_dev_;
  const uint32_t knChains, kgpuBlockSize;
};

template <typename FloatType, typename RNGType>
TRNGWrapper<FloatType, RNGType>::TRNGWrapper(const uint64_t seedNuber, const uint64_t seedDistance, const uint32_t nChains):
  knChains(nChains),
  kgpuBlockSize(CHECK_BLOCK_SIZE(1u+(nChains-1u)/NUM_THREADS_PER_BLOCK))
{
  std::vector<RNGType> rng(nChains);
  CHECK_ERROR(cudaSuccess, cudaMalloc(&rng_dev_, sizeof(RNGType)*nChains));
  for (uint32_t k=0u; k<nChains; ++k)
  {
    rng[k].seed(seedNuber);
    rng[k].jump(2u*seedDistance*k);
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
