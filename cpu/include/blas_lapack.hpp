// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <complex>
#ifdef __NVCC__
#include <cusolverDn.h>
#endif

typedef std::complex<float>  scomplex;
typedef std::complex<double> dcomplex;

extern "C"
{
  // blas routines
  void cgemv_(const char *TRANS, const int *M, const int *N, const scomplex *ALPHA,
    const scomplex *A, const int *LDA, const scomplex *X, const int *INCX,
    const scomplex *BETA, scomplex *Y, const int *INCY);
  void cgemm_(const char * TRANSA, const char * TRANSB, const int * M, const int * N, const int * K,
    const scomplex * ALPHA, const scomplex * A, const int * LDA, const scomplex * B,
    const int * LDB, const scomplex * BETA, scomplex * C, const int * LDC);
  void cgeru_(const int * M, const int * N, const scomplex * ALPHA, const scomplex * X,
    const int * INCX, const scomplex * Y, const int * INCY, scomplex * A, const int * LDA);
  void cher_(const char* UPLO, const int * N, const scomplex * ALPHA, const scomplex * X,
    const int * INCX, scomplex * A, const int * LDA);
  void cherk_(const char* UPLO, const char* TRANS, const int * N, const int * K, const scomplex * ALPHA,
    const scomplex * A, const int * LDA, const scomplex * BETA, scomplex * C, const int * LDC);
  void chemv_(const char* UPLO, const int * N, const scomplex * ALPHA, const scomplex* A, const int * LDA,
    const scomplex * X, const int * INCX, const scomplex * BETA, scomplex * Y, const int * INCY);

  void zgemv_(const char *TRANS, const int *M, const int *N, const dcomplex *ALPHA,
    const dcomplex *A, const int *LDA, const dcomplex *X, const int *INCX,
    const dcomplex *BETA, dcomplex *Y, const int *INCY);
  void zgemm_(const char * TRANSA, const char * TRANSB, const int * M, const int * N, const int * K,
    const dcomplex * ALPHA, const dcomplex * A, const int * LDA, const dcomplex * B,
    const int * LDB, const dcomplex * BETA, dcomplex * C, const int * LDC);
  void zgeru_(const int * M, const int * N, const dcomplex * ALPHA, const dcomplex * X,
    const int * INCX, const dcomplex * Y, const int * INCY, dcomplex * A, const int * LDA);
  void zher_(const char* UPLO, const int * N, const dcomplex * ALPHA, const dcomplex * X,
    const int * INCX, dcomplex * A, const int * LDA);
  void zherk_(const char* UPLO, const char* TRANS, const int * N, const int * K, const dcomplex * ALPHA,
    const dcomplex * A, const int * LDA, const dcomplex * BETA, dcomplex * C, const int * LDC);
  void zhemv_(const char* UPLO, const int * N, const dcomplex * ALPHA, const dcomplex* A, const int * LDA,
    const dcomplex * X, const int * INCX, const dcomplex * BETA, dcomplex * Y, const int * INCY);

  // lapack routines
  void chesv_(const char * UPLO, const int * N, const int * NRHS, scomplex * A,
    const int * LDA, int * IPIV, scomplex * B, const int * LDB,
    scomplex * WORK, const int * LWORK, int * INFO);
  void cgelsd_(const int * M, const int * N, const int * NRHS, const scomplex * A, const int * LDA,
    scomplex * B, const int * LDB, float * S, const float * RCOND, int * RANK,
    scomplex * WORK, const int * LWORK, float * RWORK, int * IWORK, int * INFO);

  void zhesv_(const char * UPLO, const int * N, const int * NRHS, dcomplex * A,
    const int * LDA, int * IPIV, dcomplex * B, const int * LDB,
    dcomplex * WORK, const int * LWORK, int * INFO);
  void zgelsd_(const int * M, const int * N, const int * NRHS, const dcomplex * A, const int * LDA,
    dcomplex * B, const int * LDB, double * S, const double * RCOND, int * RANK,
    dcomplex * WORK, const int * LWORK, double * RWORK, int * IWORK, int * INFO);
}

namespace blas
{
  // y = alpha*(A*X) + beta*y
  inline void gemv(const int m, const int n, const scomplex alpha,
    const scomplex *A, const scomplex *X, const scomplex beta, scomplex *y)
  {
    const char trans = 'N';
    const int inc = 1;
    cgemv_(&trans, &m, &n, &alpha, A, &m, X, &inc, &beta, y, &inc);
  }

  inline void gemv(const int m, const int n, const dcomplex alpha,
    const dcomplex *A, const dcomplex *X, const dcomplex beta, dcomplex *y)
  {
    const char trans = 'N';
    const int inc = 1;
    zgemv_(&trans, &m, &n, &alpha, A, &m, X, &inc, &beta, y, &inc);
  }


  inline void gemm(const int m, const int n, const int k,
    const scomplex & alpha, const scomplex & beta,
    const scomplex * A, const scomplex * B, scomplex * C)
  {
    const char trans = 'N';
    const int lda = m, ldb = k, ldc = m;
    cgemm_(&trans, &trans, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }

  inline void gemm(const int m, const int n, const int k,
    const dcomplex & alpha, const dcomplex & beta,
    const dcomplex * A, const dcomplex * B, dcomplex * C)
  {
    const char trans = 'N';
    const int lda = m, ldb = k, ldc = m;
    zgemm_(&trans, &trans, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }


  inline void ger(const int m, const int n, const scomplex & alpha,
    const scomplex * x, const scomplex * y, scomplex * z)
  {
    const int inc = 1;
    cgeru_(&m, &n, &alpha, x, &inc, y, &inc, z, &m);
  }

  inline void ger(const int m, const int n, const dcomplex & alpha,
    const dcomplex * x, const dcomplex * y, dcomplex * z)
  {
    const int inc = 1;
    zgeru_(&m, &n, &alpha, x, &inc, y, &inc, z, &m);
  }


  inline void her(const int n, const scomplex & alpha, const scomplex * x, scomplex * A)
  {
    const char uplo = 'L';
    const int inc = 1;
    cher_(&uplo, &n, &alpha, x, &inc, A, &n);
  }

  inline void her(const int n, const dcomplex & alpha, const dcomplex * x, dcomplex * A)
  {
    const char uplo = 'L';
    const int inc = 1;
    zher_(&uplo, &n, &alpha, x, &inc, A, &n);
  }


  inline void herk(const int n, const int k, const scomplex & alpha,
    const scomplex * A, const scomplex & beta, scomplex * C)
  {
    const char uplo = 'L';
    const char trans = 'N';
    cherk_(&uplo, &trans, &n, &k, &alpha, A, &n, &beta, C, &n);
  }

  inline void herk(const int n, const int k, const dcomplex & alpha,
    const dcomplex * A, const dcomplex & beta, dcomplex * C)
  {
    const char uplo = 'L';
    const char trans = 'N';
    zherk_(&uplo, &trans, &n, &k, &alpha, A, &n, &beta, C, &n);
  }


  inline void hemv(const int n, const scomplex alpha, const scomplex *a,
    const scomplex *x, const scomplex beta, scomplex *y)
  {
    const char uplo = 'L';
    const int inc = 1;
    chemv_(&uplo, &n, &alpha, a, &n, x, &inc, &beta, y, &inc);
  }

  inline void hemv(const int n, const dcomplex alpha, const dcomplex *a,
    const dcomplex *x, const dcomplex beta, dcomplex *y)
  {
    const char uplo = 'L';
    const int inc = 1;
    zhemv_(&uplo, &n, &alpha, a, &n, x, &inc, &beta, y, &inc);
  }
}

namespace lapack
{
  inline void sysv(const char * uplo, const int * n, scomplex * A, int * ipiv,
    scomplex * B, scomplex * work, const int * lwork, int * info)
  {
    const int nrhs = 1;
    chesv_(uplo, n, &nrhs, A, n, ipiv, B, n, work, lwork, info);
  }

  inline void sysv(const char * uplo, const int * n, dcomplex * A, int * ipiv,
    dcomplex * B, dcomplex * work, const int * lwork, int * info)
  {
    const int nrhs = 1;
    zhesv_(uplo, n, &nrhs, A, n, ipiv, B, n, work, lwork, info);
  }


  inline void gelsd(const int m, const int n, const scomplex * A, scomplex * B, float * S, const float rcond,
    int * rank, scomplex * work, const int lwork, float * rwork, int * iwork, int * info)
  {
    const int nrhs = 1;
    const int ldb = ((m>=n) ? m : n);
    cgelsd_(&m, &n, &nrhs, A, &m, B, &ldb, S, &rcond, rank, work, &lwork, rwork, iwork, info);
  }

  inline void gelsd(const int m, const int n, const dcomplex * A, dcomplex * B, double * S, const double rcond,
    int * rank, dcomplex * work, const int lwork, double * rwork, int * iwork, int * info)
  {
    const int nrhs = 1;
    const int ldb = ((m>=n) ? m : n);
    zgelsd_(&m, &n, &nrhs, A, &m, B, &ldb, S, &rcond, rank, work, &lwork, rwork, iwork, info);
  }
}

#ifdef __NVCC__
namespace cusolver
{
inline cusolverStatus_t cusolverDnTgetrf_bufferSize(const cusolverDnHandle_t & cusolverH,
  const int & n, const scomplex * A_dev, int * lwork)
{
  return cusolverDnCgetrf_bufferSize(cusolverH, n, n, (cuFloatComplex*)A_dev, n, lwork);
}

inline cusolverStatus_t cusolverDnTgetrf_bufferSize(const cusolverDnHandle_t & cusolverH,
  const int n, dcomplex * A_dev, int * lwork)
{
  return cusolverDnZgetrf_bufferSize(cusolverH, n, n, (cuDoubleComplex*)A_dev, n, lwork);
}

inline cusolverStatus_t cusolverDnTgetrf(const cusolverDnHandle_t & cusolverH, const int n,
  scomplex * A_dev, scomplex * workspace_dev, int * ipiv_dev, int * info_dev)
{
  return cusolverDnCgetrf(cusolverH, n, n, (cuFloatComplex*)A_dev, n,
    (cuFloatComplex*)workspace_dev, ipiv_dev, info_dev);
}

inline cusolverStatus_t cusolverDnTgetrf(const cusolverDnHandle_t & cusolverH, const int n,
  dcomplex * A_dev, dcomplex * workspace_dev, int * ipiv_dev, int * info_dev)
{
  return cusolverDnZgetrf(cusolverH, n, n, (cuDoubleComplex*)A_dev, n,
    (cuDoubleComplex*)workspace_dev, ipiv_dev, info_dev);
}

inline cusolverStatus_t cusolverDnTgetrs(const cusolverDnHandle_t & cusolverH, const int n,
  scomplex * A_dev, const int * ipiv_dev, scomplex * B_dev, int * info_dev)
{
  return cusolverDnCgetrs(cusolverH, CUBLAS_OP_N, n, 1,
    (cuFloatComplex*)A_dev, n, ipiv_dev, (cuFloatComplex*)B_dev, n, info_dev);
}

inline cusolverStatus_t cusolverDnTgetrs(const cusolverDnHandle_t & cusolverH, const int n,
  dcomplex * A_dev, const int * ipiv_dev, dcomplex * B_dev, int * info_dev)
{
  return cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, n, 1,
    (cuDoubleComplex*)A_dev, n, ipiv_dev, (cuDoubleComplex*)B_dev, n, info_dev);
}
} // namespace cusolver
#endif
