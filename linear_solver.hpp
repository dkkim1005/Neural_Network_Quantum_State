#pragma once

#include <vector>
#include <exception>
#include <cmath>
#include "blas_lapack.hpp"

template <typename float_t>
class LinearSolver
{
public:
  explicit LinearSolver(const int n): kn(n), ipiv_(n, 0) {}
  void solve(std::complex<float_t> * A, std::complex<float_t> * B, const char uplo)
  {
    int info = 1, lwork = -1;
    std::vector<std::complex<float_t> > work(1);
    // query mode (lwork = -1, finding an optimal workspace)
    lapack::sysv(&uplo, &kn, A, &ipiv_[0], B, &work[0], &lwork, &info);
    lwork = static_cast<int>(work[0].real());
    std::vector<std::complex<float_t> >(lwork).swap(work);
    // solve linear eq Ax = B for x
    lapack::sysv(&uplo, &kn, A, &ipiv_[0], B, &work[0], &lwork, &info);
    if (info)
      throw std::runtime_error(" @Error! No solutions for the linear eq(A*x = B). ");
  }
private:
  const int kn;
  std::vector<int> ipiv_;
};


template <typename float_t>
class PsuedoInverseSolver
{
public:
  PsuedoInverseSolver(const int m, const int n);
  void solve(std::complex<float_t> * A, std::complex<float_t> * B, const float_t rcond)
  {
    int rank, lwork, info;
    // query mode (lwork = -1, finding an optimal workspace)
    lwork = -1;
    std::vector<std::complex<float_t> > work(1);
    lapack::gelsd(km, kn, A, B, &S_[0], rcond, &rank, &work[0], lwork, &rwork_[0], &iwork_[0], &info);
    lwork = static_cast<int>(work[0].real());
    std::vector<std::complex<float_t> >(lwork).swap(work);
    // solve the linear least square: min_x |Ax-B|_2
    lapack::gelsd(km, kn, A, B, &S_[0], rcond, &rank, &work[0], lwork, &rwork_[0], &iwork_[0], &info);
    if (info)
      throw std::runtime_error(" @Error! computing SVD failed to converge...");
  }
private:
  const int km, kn, knlvl;
  std::vector<int> iwork_;
  std::vector<float_t> S_, rwork_;
};

template <typename float_t>
PsuedoInverseSolver<float_t>::PsuedoInverseSolver(const int m, const int n):
km(m),
kn(n),
knlvl(static_cast<int>(std::log2(((m>n)?n:m)/26.0))+1)
{
  const int minmn = ((m>n)?n:m), liwork = 3*minmn*knlvl + 11*minmn, nrhs = 1, lrwork = 10*n+2*n*25+8*n*knlvl+3*25*nrhs+26*26;
  iwork_.resize(liwork);
  rwork_.resize(lrwork);
  S_.resize(((m>n)?n:m));
}
