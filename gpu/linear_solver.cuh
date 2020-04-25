#pragma once

#include <thrust/complex.h>
#include <string>
#include <exception>
#include <cublas_v2.h>
#include <magma.h>
#include <magma_lapack.h>
#include "common.cuh"

namespace linearsolver
{
template <typename FloatType> class cudaBKF;
template <> class cudaBKF<double>
{
public:
  explicit cudaBKF(const int n, const bool hasInitializedMagma = false):
    kHasInitializedMagma(hasInitializedMagma),
    kn(n)
  {
    if (!hasInitializedMagma)
      CHECK_ERROR(MAGMA_SUCCESS, magma_init());
    CHECK_ERROR(cudaSuccess, cudaMalloc(&dworks_dev_, sizeof(thrust::complex<float>)*n*(n+1)));
    CHECK_ERROR(cudaSuccess, cudaMalloc(&dworkd_dev_, sizeof(thrust::complex<double>)*n));
  }

  ~cudaBKF()
  {
    CHECK_ERROR(cudaSuccess, cudaFree(dworks_dev_));
    CHECK_ERROR(cudaSuccess, cudaFree(dworkd_dev_));
    if (!kHasInitializedMagma)
      CHECK_ERROR(MAGMA_SUCCESS, magma_finalize());
  }

  void solve(thrust::complex<double> * A_dev, thrust::complex<double> * B_dev, thrust::complex<double> * x_dev, magma_uplo_t uplo = MagmaLower)
  {
    magma_int_t iter, info;
    CHECK_ERROR(MAGMA_SUCCESS, magma_zchesv_gpu(uplo, kn, 1,
      reinterpret_cast<magmaDoubleComplex*>(A_dev), kn,
      reinterpret_cast<magmaDoubleComplex*>(B_dev), kn,
      reinterpret_cast<magmaDoubleComplex*>(x_dev), kn,
      reinterpret_cast<magmaDoubleComplex*>(dworkd_dev_),
      reinterpret_cast<magmaFloatComplex*>(dworks_dev_), &iter, &info));
    if (info < 0)
      throw std::runtime_error("output of magma_zchesv_gpu: info=" + std::to_string(info));
  }

private:
  const bool kHasInitializedMagma;
  const int kn;
  thrust::complex<float> * dworks_dev_;
  thrust::complex<double> * dworkd_dev_;
};

template <typename FloatType>
class cudaCF
{
public:
  cudaCF(const int n, const bool hasInitializedMagma = false):
    kHasInitializedMagma(hasInitializedMagma),
    kn(n)
  {
    if (!hasInitializedMagma)
      CHECK_ERROR(MAGMA_SUCCESS, magma_init());
  }

  ~cudaCF()
  {
    if (!kHasInitializedMagma)
      CHECK_ERROR(MAGMA_SUCCESS, magma_finalize());
  }

  void solve(thrust::complex<FloatType> * A_dev, thrust::complex<FloatType> * B_dev, thrust::complex<FloatType> * x_dev, magma_uplo_t uplo = MagmaLower);

private:
  const bool kHasInitializedMagma;
  const int kn;
};

template <>
void cudaCF<float>::solve(thrust::complex<float> * A_dev, thrust::complex<float> * B_dev, thrust::complex<float> * x_dev, magma_uplo_t uplo)
{
  magma_int_t info;
  CHECK_ERROR(MAGMA_SUCCESS, magma_cposv_gpu(uplo, kn, 1,
    reinterpret_cast<magmaFloatComplex*>(A_dev), kn,
    reinterpret_cast<magmaFloatComplex*>(B_dev), kn, &info));
  if (info < 0)
    throw std::runtime_error("output of magma_cposv_gpu: info=" + std::to_string(info));
  CHECK_ERROR(cudaSuccess, cudaMemcpy(x_dev, B_dev, sizeof(thrust::complex<float>)*kn, cudaMemcpyDeviceToDevice));
}

template <>
void cudaCF<double>::solve(thrust::complex<double> * A_dev, thrust::complex<double> * B_dev, thrust::complex<double> * x_dev, magma_uplo_t uplo)
{
  magma_int_t info;
  CHECK_ERROR(MAGMA_SUCCESS, magma_zposv_gpu(uplo, kn, 1,
    reinterpret_cast<magmaDoubleComplex*>(A_dev), kn,
    reinterpret_cast<magmaDoubleComplex*>(B_dev), kn, &info));
  if (info < 0)
    throw std::runtime_error("output of magma_zposv_gpu: info=" + std::to_string(info));
  CHECK_ERROR(cudaSuccess, cudaMemcpy(x_dev, B_dev, sizeof(thrust::complex<double>)*kn, cudaMemcpyDeviceToDevice));
}
} // namespace linearsolver
