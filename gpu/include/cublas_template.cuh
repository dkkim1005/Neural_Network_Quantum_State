// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <cublas_v2.h>
#include <thrust/complex.h>

#define SCOMPLEX_FROM_THRUST_TO_CUDA_PTR(DEVICE_COMPLEX_PTR)\
reinterpret_cast<cuComplex*>(DEVICE_COMPLEX_PTR)
#define DCOMPLEX_FROM_THRUST_TO_CUDA_PTR(DEVICE_COMPLEX_PTR)\
reinterpret_cast<cuDoubleComplex*>(DEVICE_COMPLEX_PTR)
#define SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(DEVICE_COMPLEX_PTR)\
reinterpret_cast<const cuComplex*>(DEVICE_COMPLEX_PTR)
#define DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(DEVICE_COMPLEX_PTR)\
reinterpret_cast<const cuDoubleComplex*>(DEVICE_COMPLEX_PTR)

namespace cublas
{
// y = alpha*(A*X) + beta*y
inline void gemv(const cublasHandle_t & handle, const int & m, const int & n, const thrust::complex<float> & alpha,
  const thrust::complex<float> * A, const thrust::complex<float> * x, const thrust::complex<float> & beta, thrust::complex<float> * y)
{
  const cublasOperation_t trans = CUBLAS_OP_N;
  cublasCgemv(handle, trans, m, n,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&alpha),
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(A), m,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(x), 1,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&beta),
    SCOMPLEX_FROM_THRUST_TO_CUDA_PTR(y), 1);
}

inline void gemv(const cublasHandle_t & handle, const int & m, const int & n, const thrust::complex<double> & alpha,
  const thrust::complex<double> * A, const thrust::complex<double> * x, const thrust::complex<double> & beta, thrust::complex<double> * y)
{
  const cublasOperation_t trans = CUBLAS_OP_N;
  cublasZgemv(handle, trans, m, n,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&alpha),
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(A), m,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(x), 1,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&beta),
    DCOMPLEX_FROM_THRUST_TO_CUDA_PTR(y), 1);
}

// C = alpha*(A*B) + beta*C
inline void gemm(const cublasHandle_t & handle, const int & m, const int & n, const int & k, const thrust::complex<float> & alpha,
  const thrust::complex<float> & beta, const thrust::complex<float> * A, const thrust::complex<float> * B, thrust::complex<float> * C)
{
  const int lda = m, ldb = k, ldc = m;
  const cublasOperation_t trans = CUBLAS_OP_N;
  cublasCgemm(handle, trans, trans, m, n, k,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&alpha),
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(A), lda,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(B), ldb,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&beta),
    SCOMPLEX_FROM_THRUST_TO_CUDA_PTR(C), ldc);
}

inline void gemm(const cublasHandle_t & handle, const int & m, const int & n, const int & k, const thrust::complex<double> & alpha,
  const thrust::complex<double> & beta, const thrust::complex<double> * A, const thrust::complex<double> * B, thrust::complex<double> * C)
{
  const int lda = m, ldb = k, ldc = m;
  const cublasOperation_t trans = CUBLAS_OP_N;
  cublasZgemm(handle, trans, trans, m, n, k,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&alpha),
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(A), lda,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(B), ldb,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&beta),
    DCOMPLEX_FROM_THRUST_TO_CUDA_PTR(C), ldc);
}

// z = alpha*(x*y**T) + z
inline void ger(const cublasHandle_t & handle, const int & m, const int & n, const thrust::complex<float> & alpha,
  const thrust::complex<float> * x, const thrust::complex<float> * y, thrust::complex<float> * z)
{
  const int inc = 1;
  cublasCgeru(handle, m, n,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&alpha),
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(x), inc,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(y), inc,
    SCOMPLEX_FROM_THRUST_TO_CUDA_PTR(z), m);
}

inline void ger(const cublasHandle_t & handle, const int & m, const int & n, const thrust::complex<double> & alpha,
  const thrust::complex<double> * x, const thrust::complex<double> * y, thrust::complex<double> * z)
{
  const int inc = 1;
  cublasZgeru(handle, m, n,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&alpha),
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(x), inc,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(y), inc,
    DCOMPLEX_FROM_THRUST_TO_CUDA_PTR(z), m);
}

// A = alpha*x*x**H + A
inline void her(const cublasHandle_t & handle, const int & n, const float & alpha, const thrust::complex<float> * x, thrust::complex<float> * A)
{
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const int inc = 1;
  cublasCher(handle, uplo, n, &alpha,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(x), inc,
    SCOMPLEX_FROM_THRUST_TO_CUDA_PTR(A), n);
}

inline void her(const cublasHandle_t & handle, const int & n, const double & alpha, const thrust::complex<double> * x, thrust::complex<double> * A)
{
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const int inc = 1;
  cublasZher(handle, uplo, n, &alpha,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(x), inc,
    DCOMPLEX_FROM_THRUST_TO_CUDA_PTR(A), n);
}

// C = alpha*A*A**H + beta*C
inline void herk(const cublasHandle_t & handle, const int & n, const int & k, const float & alpha,
  const thrust::complex<float> * A, const float & beta, thrust::complex<float> * C)
{
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const cublasOperation_t trans = CUBLAS_OP_N;
  cublasCherk(handle, uplo, trans, n, k, &alpha,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(A), n, &beta,
    SCOMPLEX_FROM_THRUST_TO_CUDA_PTR(C), n);
}

inline void herk(const cublasHandle_t & handle, const int & n, const int & k, const double & alpha,
  const thrust::complex<double> * A, const double & beta, thrust::complex<double> * C)
{
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const cublasOperation_t trans = CUBLAS_OP_N;
  cublasZherk(handle, uplo, trans, n, k, &alpha,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(A), n, &beta,
    DCOMPLEX_FROM_THRUST_TO_CUDA_PTR(C), n);
}

// y = alpha*A*x + beta*y
inline void hemv(const cublasHandle_t & handle, const int & n, const thrust::complex<float> & alpha, const thrust::complex<float> * A,
  const thrust::complex<float> * x, const thrust::complex<float> & beta, thrust::complex<float> * y)
{
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const int inc = 1;
  cublasChemv(handle, uplo, n,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&alpha),
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(A), n,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(x), inc,
    SCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&beta),
    SCOMPLEX_FROM_THRUST_TO_CUDA_PTR(y), inc);
}

inline void hemv(const cublasHandle_t & handle, const int & n, const thrust::complex<double> & alpha, const thrust::complex<double> * A,
  const thrust::complex<double> * x, const thrust::complex<double> & beta, thrust::complex<double> * y)
{
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const int inc = 1;
  cublasZhemv(handle, uplo, n,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&alpha),
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(A), n,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(x), inc,
    DCOMPLEX_FROM_THRUST_TO_CUDA_CONST_PTR(&beta),
    DCOMPLEX_FROM_THRUST_TO_CUDA_PTR(y), inc);
}
} // namespace cublas
