// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iostream>
#include <cmath>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

namespace internal_impl
{
template <typename FloatType>
struct NormFunctor
{
  __host__ __device__ FloatType operator()(const thrust::complex<FloatType> & x) const
  {
    return thrust::norm(x);
  }
};

template <typename FloatType>
struct aDotbFunctor: public thrust::binary_function<thrust::complex<FloatType>,
  thrust::complex<FloatType>, thrust::complex<FloatType>>
{
  __host__ __device__ thrust::complex<FloatType> operator()(thrust::complex<FloatType> a, thrust::complex<FloatType> b) const
  {
    return a*thrust::conj(b);
  }
};

template <typename FloatType>
struct AxpyFunctor: public thrust::binary_function<thrust::complex<FloatType>, thrust::complex<FloatType>, thrust::complex<FloatType>>
{
  AxpyFunctor(const thrust::complex<FloatType> alpha, const thrust::complex<FloatType> beta = 1):
    kalpha(alpha),
    kbeta(beta)
  {}

  __host__ __device__ thrust::complex<FloatType> operator()(const thrust::complex<FloatType> & a, const thrust::complex<FloatType> & b) const
  {
    return (kalpha*a)+(kbeta*b);
  }

private:
  const thrust::complex<FloatType> kalpha, kbeta;
};

// return \sum_i |v_i|^2
template <typename FloatType>
FloatType squared_norm(const thrust::device_vector<thrust::complex<FloatType>> & v)
{
  const FloatType zero = 0;
  return thrust::transform_reduce(v.begin(), v.end(), NormFunctor<FloatType>(), zero, thrust::plus<FloatType>());
}

// return \sum_i (conj(a_i)*b_i)
template <typename FloatType>
thrust::complex<FloatType> hermition_inner_product(const thrust::device_vector<thrust::complex<FloatType>> & a,
  const thrust::device_vector<thrust::complex<FloatType>> & b)
{
  const thrust::complex<FloatType> kzero = thrust::complex<FloatType>(0, 0);
  return thrust::inner_product(a.begin(), a.end(), b.begin(), kzero, thrust::plus<thrust::complex<FloatType>>(), aDotbFunctor<FloatType>());
}
} // internal_impl


/*
  GPU-ConjugateGradient for solving a Hermition matrix 
  This code is designed to run a GPU machine without cpu-gpu memory transfer of arrays.
  I refer to "ConjugateGradient.h" of Eigen3 library, so that the most parts of the code follows the same logic.
  (http://eigen.tuxfamily.org/dox/)
*/
template <typename FloatType>
class ConjugateGradient
{
public:
  ConjugateGradient(const int n, const FloatType tolerance, const int maxIter = 1000):
    kmaxIters(maxIter),
    ktol(tolerance),
    tmp_dev_(n),
    residual_dev_(n),
    p_dev_(n),
    z_dev_(n),
    kconsiderAsZero(std::numeric_limits<FloatType>::min())
  {}

  template <typename MatrixInterface>
  void solve(MatrixInterface & Mat, const thrust::device_vector<thrust::complex<FloatType>> & rhs_dev,
    thrust::device_vector<thrust::complex<FloatType>> & x_dev)
  {
    // residual = rhs - Mat.dot(x)
    Mat.dot(x_dev, tmp_dev_);
    thrust::transform(rhs_dev.begin(), rhs_dev.end(), tmp_dev_.begin(), residual_dev_.begin(), thrust::minus<thrust::complex<FloatType>>());
    //rhsNorm2 = L2Norm(Mat.dot(rhs))
    FloatType rhsNorm2 = internal_impl::squared_norm(rhs_dev);
    if (rhsNorm2 == 0)
    {
      thrust::fill(x_dev.begin(), x_dev.end(), thrust::complex<FloatType>(0, 0));
      return;
    }
    const FloatType threshold = std::max(ktol*ktol*rhsNorm2, kconsiderAsZero);
    FloatType residualNorm2 = internal_impl::squared_norm(residual_dev_);
    if (residualNorm2 < threshold)
      return;
    Mat.applyPrecond(residual_dev_, p_dev_);
    FloatType abs[2]; // abs[0] : absOld, abs[1] : absNew
    abs[1] = (internal_impl::hermition_inner_product(p_dev_, residual_dev_)).real();
    for (int iter=0; iter<kmaxIters; ++iter)
    {
      // the bottleneck of the algorithm
      Mat.dot(p_dev_, tmp_dev_);
      // the amount we travel on dir
      const FloatType alpha = abs[1]/(internal_impl::hermition_inner_product(tmp_dev_, p_dev_)).real();
      // update solution (x += alpha*p)
      thrust::transform(p_dev_.begin(), p_dev_.end(), x_dev.begin(), x_dev.begin(), internal_impl::AxpyFunctor<double>(alpha));
      // update residual (residual -= alpha*tmp)
      thrust::transform(tmp_dev_.begin(), tmp_dev_.end(), residual_dev_.begin(), residual_dev_.begin(), internal_impl::AxpyFunctor<double>(-alpha));
      residualNorm2 = internal_impl::squared_norm(residual_dev_);
      if (residualNorm2 < threshold)
        break;
      // approximately solve for "A z = residual"
      Mat.applyPrecond(residual_dev_, z_dev_);
      abs[0] = abs[1];
      // update the absolute value of r
      abs[1] = (internal_impl::hermition_inner_product(z_dev_, residual_dev_)).real();
      // calculate the Gram-Schmidt value used to create the new search direction
      const FloatType beta = abs[1]/abs[0];
      // update search direction (p = z + beta*p)
      thrust::transform(z_dev_.begin(), z_dev_.end(), p_dev_.begin(), p_dev_.begin(), internal_impl::AxpyFunctor<double>(1, beta));
    }
    //std::cout << "# tol_error: " << std::sqrt(residualNorm2/rhsNorm2) << std::endl;
  }

private:
  const int kmaxIters;
  const FloatType ktol, kconsiderAsZero;
  thrust::device_vector<thrust::complex<FloatType>> tmp_dev_, residual_dev_, p_dev_, z_dev_;
};
