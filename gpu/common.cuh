// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <stdio.h>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <cublas_v2.h>

#define CHECK_ERROR(SUCCESS_MACRO, STATUS) do {\
  if (SUCCESS_MACRO != (STATUS)) {\
    std::cerr << "# ERROR --- FILE:" << __FILE__ << ", LINE:" << __LINE__ << std::endl;\
    exit(1);\
  }\
} while (false)

#define CHECK_BLOCK_SIZE(x) ((x<65535u) ? x : 65535u)
#define NUM_THREADS_PER_BLOCK 32u
#define PTR_FROM_THRUST(THRUST_DEVICE_PTR) thrust::raw_pointer_cast(THRUST_DEVICE_PTR)

namespace gpu_kernel
{
template <typename FloatType>
__global__ void common__Print__(const thrust::complex<FloatType> * A, const uint32_t nrow, const uint32_t ncol)
{
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx == 0u)
    for (uint32_t i=0u; i<nrow; ++i)
    {
      for (uint32_t j=0u; j<ncol; ++j)
        printf("(%.8e,%.8e) ", A[i*ncol+j].real(), A[i*ncol+j].imag());
      printf("\n");
    }
}

template <typename FloatType>
__global__ void common__Print__(const FloatType * A, const uint32_t nrow, const uint32_t ncol)
{
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx == 0u)
    for (uint32_t i=0u; i<nrow; ++i)
    {
      for (uint32_t j=0u; j<ncol; ++j)
        printf("%f ", A[i*ncol+j]);
      printf("\n");
    }
}

template <typename T1, typename T2>
__global__ void common__SetValues__(T1 * A, const uint32_t size, const T2 value)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < size)
  {
    A[idx] = value;
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void common__ApplyComplexConjugateVector__(thrust::complex<FloatType> * vec, const uint32_t size)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < size)
  {
    vec[idx] = thrust::conj(vec[idx]);
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void common__copyFromRealToImag__(const FloatType * real, const uint32_t size, thrust::complex<FloatType> * imag)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < size)
  {
    imag[idx] = real[idx];
    idx += nstep;
  }
}
} // namespace gpu_kernel
