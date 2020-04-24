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
} // namespace linearsolver
