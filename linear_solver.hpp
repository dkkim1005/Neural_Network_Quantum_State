// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <vector>
#include <exception>
#include <cmath>
#include "blas_lapack.hpp"
#include "minresqlp.hpp"

namespace linearsolver
{
// Solve Ax=b (Bunch-Kaufman factorization)
template <typename FloatType>
class BKF
{
public:
  explicit BKF(const int n);
  void solve(std::complex<FloatType> * A, std::complex<FloatType> * B);
private:
  const int kn;
  std::vector<int> ipiv_;
};

// Minimize |Ax-b|^2 (SVD)
template <typename FloatType>
class SVD
{
public:
  explicit SVD(const int n);
  void solve(std::complex<FloatType> * A, std::complex<FloatType> * B);
private:
  const int km, kn, knlvl;
  std::vector<int> iwork_;
  std::vector<FloatType> S_, rwork_;
};

// Minimize |x| among the solutions of the psuedo inverse problem for Ax=b (minresqlp)
template <typename FloatType>
class MINRESQLP
{
public:
  explicit MINRESQLP(const int n);
  void solve(std::complex<FloatType> * A, std::complex<FloatType> * B);
private:
  class HermitianOP_: public minresqlp::BaseInterface<HermitianOP_, IMAG<FloatType> >
  {
  public:
    HermitianOP_(const int n, const std::complex<FloatType> * b, const std::complex<FloatType> * A);
    void Aprod(const int n, const std::complex<FloatType> *x, std::complex<FloatType> *y) const;
  private:
    const std::complex<FloatType> * A_;
    const std::complex<FloatType> kone, kzero;
  };
  const int kn;
  minresqlp::HermitianSolver<HermitianOP_, FloatType> solver_;
};


/*== implementation of bfk class ==*/
template <typename FloatType>
BKF<FloatType>::BKF(const int n):
  kn(n),
  ipiv_(n, 0) {}

template <typename FloatType>
void BKF<FloatType>::solve(std::complex<FloatType> * A, std::complex<FloatType> * B)
{
  const char uplo = 'L';
  int info = 1, lwork = -1;
  std::vector<std::complex<FloatType> > work(1);
  // query mode (lwork = -1, finding an optimal workspace)
  lapack::sysv(&uplo, &kn, A, &ipiv_[0], B, &work[0], &lwork, &info);
  lwork = static_cast<int>(work[0].real());
  std::vector<std::complex<FloatType> >(lwork).swap(work);
  // solve linear eq Ax = B for x
  lapack::sysv(&uplo, &kn, A, &ipiv_[0], B, &work[0], &lwork, &info);
  if (info)
    throw std::runtime_error("No solutions for the linear eq(A*x = B).");
}


/*== implementation of SVD class ==*/
template <typename FloatType>
SVD<FloatType>::SVD(const int n):
  km(n),
  kn(n),
  knlvl(static_cast<int>(std::log2((n/26.0))+1))
{
  const int minmn = ((km>kn)?kn:km), liwork = 3*minmn*knlvl + 11*minmn, nrhs = 1, lrwork = 10*kn+2*kn*25+8*kn*knlvl+3*25*nrhs+26*26;
  iwork_.resize(liwork);
  rwork_.resize(lrwork);
  S_.resize(((km>kn)?kn:km));
}

template <typename FloatType>
void SVD<FloatType>::solve(std::complex<FloatType> * A, std::complex<FloatType> * B)
{
  int rank, lwork, info;
  // query mode (lwork = -1, finding an optimal workspace)
  lwork = -1;
  FloatType rcond = 1e-10;
  std::vector<std::complex<FloatType> > work(1);
  lapack::gelsd(km, kn, A, B, &S_[0], rcond, &rank, &work[0], lwork, &rwork_[0], &iwork_[0], &info);
  lwork = static_cast<int>(work[0].real());
  std::vector<std::complex<FloatType> >(lwork).swap(work);
  // solve the linear least square: min_x |Ax-B|_2
  lapack::gelsd(km, kn, A, B, &S_[0], rcond, &rank, &work[0], lwork, &rwork_[0], &iwork_[0], &info);
  if (info)
    throw std::runtime_error("Computing SVD failed to converge...");
}


/*== implementation of MINRESQLP class ==*/
template <typename FloatType>
MINRESQLP<FloatType>::MINRESQLP(const int n):
  kn(n) {}

template <typename FloatType>
void MINRESQLP<FloatType>::solve(std::complex<FloatType> * A, std::complex<FloatType> * B)
{
  HermitianOP_ zclient(kn, B, A);
  zclient.rtol = 1e-9;
  solver_.solve(zclient);
  std::memcpy(B, zclient.x.data(), sizeof(std::complex<FloatType>)*kn);
}

template <typename FloatType>
MINRESQLP<FloatType>::HermitianOP_::HermitianOP_ (const int n, const std::complex<FloatType> * b, const std::complex<FloatType> * A):
  minresqlp::BaseInterface<HermitianOP_, IMAG<FloatType> >(n, b),
  A_(A),
  kone(std::complex<FloatType>(1.0, 0.0)),
  kzero(std::complex<FloatType>(0.0, 0.0)) {}

template <typename FloatType>
void MINRESQLP<FloatType>::HermitianOP_::Aprod(const int n, const std::complex<FloatType> *x, std::complex<FloatType> *y) const
{
  blas::hemv(n, kone, A_, x, kzero, y);
}
}
