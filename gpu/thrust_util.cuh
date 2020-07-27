// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

namespace internal_impl
{
template <typename FloatType>
struct L2NormFunctor
{
  __host__ __device__ FloatType operator()(const thrust::complex<FloatType> & x) const
  {
    return thrust::norm(x);
  }
};

template <typename FloatType>
struct L4NormFunctor
{
  __host__ __device__ FloatType operator()(const FloatType & x) const
  {
    const FloatType x2 = x*x;
    return x2*x2;
  }
};

template <typename FloatType>
struct ComplexABSFunctor
{
  __host__ __device__ FloatType operator()(const thrust::complex<FloatType> & x) const
  {
    return thrust::abs(x);
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
FloatType l2_norm(const thrust::device_vector<thrust::complex<FloatType>> & v)
{
  const FloatType zero = 0;
  return thrust::transform_reduce(v.begin(), v.end(), L2NormFunctor<FloatType>(), zero, thrust::plus<FloatType>());
}

// return \sum_i |v_i|^2
template <typename FloatType>
FloatType l4_norm(const thrust::device_vector<FloatType> & v)
{
  const FloatType zero = 0;
  return thrust::transform_reduce(v.begin(), v.end(), L4NormFunctor<FloatType>(), zero, thrust::plus<FloatType>());
}

// return \sum_i (a_i*conj(b_i))
template <typename FloatType>
thrust::complex<FloatType> hermition_inner_product(const thrust::device_vector<thrust::complex<FloatType>> & a,
  const thrust::device_vector<thrust::complex<FloatType>> & b)
{
  const thrust::complex<FloatType> kzero = thrust::complex<FloatType>(0, 0);
  return thrust::inner_product(a.begin(), a.end(), b.begin(), kzero, thrust::plus<thrust::complex<FloatType>>(), aDotbFunctor<FloatType>());
}
} // internal_impl
