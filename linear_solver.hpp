#pragma once

#include "blas.hpp"
#include <vector>
#include <exception>

template <typename float_t>
class LinearSolver
{
public:
  explicit LinearSolver(const int n): kn(n), ipiv_(n, 0) {}
  void solve(std::complex<float_t> * A, std::complex<float_t> * B, const char uplo)
  {
    int info = 1, lwork = -1;
    std::vector<std::complex<float_t> > work(1);
    // query mode(lwork = -1)
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
