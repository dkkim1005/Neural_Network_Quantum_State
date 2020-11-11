#pragma once

#include <vector>
#include <complex>
#include <limits>
#include <cmath>

/*
  "ConjugateGradient.h" of Eigen3 library is refered in this code.
  The most parts of the code follows the same logic in the Eigen3 implementation.
  (http://eigen.tuxfamily.org/dox/)
*/

template <typename FloatType>
class ConjugateGradient
{
public:
  ConjugateGradient(const int n, const FloatType tolerance, const int maxIter = 1000):
    kn(n),
    kmaxIters(maxIter),
    ktol(tolerance),
    tmp_(n),
    residual_(n),
    p_(n),
    z_(n),
    kconsiderAsZero(std::numeric_limits<FloatType>::min())
  {}

  template <typename MatrixInterface>
  void solve(MatrixInterface & Mat, const std::complex<FloatType> * rhs,
    std::complex<FloatType> * x, const bool doPrintErr = false)
  {
    // residual = rhs - Mat.dot(x)
    Mat.dot(&x[0], &tmp_[0]);
    for (int i=0; i<kn; ++i)
      residual_[i] = rhs[i] - tmp_[i];
    FloatType rhsNorm2 = this->l2_norm_(&rhs[0]);
    if (rhsNorm2 == 0)
    {
      for (int i=0; i<kn; ++i)
        x[i] = 0;
      return;
    }
    const FloatType threshold = std::max(ktol*ktol*rhsNorm2, kconsiderAsZero);
    FloatType residualNorm2 = this->l2_norm_(&residual_[0]);
    if (residualNorm2 < threshold)
      return;
    Mat.applyPrecond(&residual_[0], &p_[0]);
    FloatType abs[2]; // abs[0] : absOld, abs[1] : absNew
    abs[1] = (this->hermition_inner_product_(&p_[0], &residual_[0])).real();
    for (int iter=0; iter<kmaxIters; ++iter)
    {
      // the bottleneck of the algorithm
      Mat.dot(&p_[0], &tmp_[0]);
      // the amount we travel on dir
      const FloatType alpha = abs[1]/(this->hermition_inner_product_(&tmp_[0], &p_[0])).real();
      // update solution (x += alpha*p)
      for (int i=0; i<kn; ++i)
        x[i] += alpha*p_[i];
      // update residual (residual -= alpha*tmp)
      for (int i=0; i<kn; ++i)
        residual_[i] -= alpha*tmp_[i];
      residualNorm2 = this->l2_norm_(&residual_[0]);
      if (residualNorm2 < threshold)
        break;
      // approximately solve for "A z = residual"
      Mat.applyPrecond(&residual_[0], &z_[0]);
      abs[0] = abs[1];
      // update the absolute value of r
      abs[1] = (this->hermition_inner_product_(&z_[0], &residual_[0])).real();
      // calculate the Gram-Schmidt value used to create the new search direction
      const FloatType beta = abs[1]/abs[0];
      // update search direction (p = z + beta*p)
      for (int i=0; i<kn; ++i)
        p_[i] = z_[i] + beta*p_[i];
    }
    if (doPrintErr)
      std::cout << "# tol_error: " << std::sqrt(residualNorm2/rhsNorm2) << std::endl;
  }

private:
  FloatType l2_norm_(const std::complex<FloatType> * z) const
  {
    FloatType res = 0;
    for (int i=0; i<kn; ++i)
      res += std::norm(z[i]);
    return res;
  }

  std::complex<FloatType> hermition_inner_product_(const std::complex<FloatType> * z1, const std::complex<FloatType> * z2) const
  {
    std::complex<FloatType> res = 0;
    for (int i=0; i<kn; ++i)
      res += z1[i]*std::conj(z2[i]);
    return res;
  }

  const int kmaxIters, kn;
  const FloatType ktol, kconsiderAsZero;
  std::vector<std::complex<FloatType> > tmp_, residual_, p_, z_;
};
