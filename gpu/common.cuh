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

#define CHECK_BLOCK_SIZE(x) ((x<65535) ? x : 65535)
#define NUM_THREADS_PER_BLOCK 32

#define SCOMPLEX_FROM_THRUST_TO_CUDA_PTR(DEVICE_PTR)\
reinterpret_cast<cuComplex*>(DEVICE_PTR)
#define DCOMPLEX_FROM_THRUST_TO_CUDA_PTR(DEVICE_PTR)\
reinterpret_cast<cuDoubleComplex*>(DEVICE_PTR)
#define SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(DEVICE_PTR)\
reinterpret_cast<const cuComplex*>(DEVICE_PTR)
#define DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(DEVICE_PTR)\
reinterpret_cast<const cuDoubleComplex*>(DEVICE_PTR)

#define PTR_FROM_THRUST(THRUST_DEVICE_PTR) thrust::raw_pointer_cast(THRUST_DEVICE_PTR)

namespace gpu_kernel
{
template <typename FloatType>
__global__ void common__Print__(const thrust::complex<FloatType> * A, const int nrow, const int ncol)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx == 0)
    for (int i=0; i<nrow; ++i)
    {
      for (int j=0; j<ncol; ++j)
        printf("(%.8e,%.8e) ", A[i*ncol+j].real(), A[i*ncol+j].imag());
      printf("\n");
    }
}

template <typename FloatType>
__global__ void common__Print__(const FloatType * A, const int nrow, const int ncol)
{
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx == 0)
    for (int i=0; i<nrow; ++i)
    {
      for (int j=0; j<ncol; ++j)
        printf("%f ", A[i*ncol+j]);
      printf("\n");
    }
}

template <typename T1, typename T2>
__global__ void common__SetValues__(T1 * A, const int size, const T2 value)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < size)
  {
    A[idx] = value;
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void common__ApplyComplexConjugateVector__(thrust::complex<FloatType> * vec, const int size)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < size)
  {
    vec[idx] = thrust::conj(vec[idx]);
    idx += nstep;
  }
}
} // namespace cuda_common

// implementation for the circular list structure
template <typename FloatType = int>
class OneWayLinkedIndex
{
public:
  void set_item(const FloatType & item) { item_ = item; }
  void set_nextptr(OneWayLinkedIndex * nextPtr) { nextPtr_ = nextPtr; }
  OneWayLinkedIndex * next_ptr() const { return nextPtr_; }
  FloatType get_item() { return item_; }
private:
  FloatType item_;
  OneWayLinkedIndex * nextPtr_;
};
