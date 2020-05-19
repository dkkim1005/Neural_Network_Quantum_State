// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <exception>
#include "conjugate_gradient.cuh"
#include "cublas_template.cuh"
#include "common.cuh"

template <typename FloatType>
class SMatrixForCG
{
public:
  SMatrixForCG(const int nChains, const int nVariables);
  ~SMatrixForCG();
  // (lambda): S = S + \lambda*diag(S)
  void set_lnpsiGradients(const thrust::device_vector<thrust::complex<FloatType>> & lnpsiGradients_dev, const FloatType lambda = 0);
  // b = S*a
  void dot(const thrust::device_vector<thrust::complex<FloatType>> & a, thrust::device_vector<thrust::complex<FloatType>> & b);
  // solve M*a = b where M is the preconditioner
  void applyPrecond(const thrust::device_vector<thrust::complex<FloatType>> & rhs_dev, thrust::device_vector<thrust::complex<FloatType>> & x_dev) const;
private:
  const int knChains, knVariables, kgpuBlockSize1, kgpuBlockSize2;
  const thrust::complex<FloatType> kone, kzero, kmone, kinvNchains;
  const thrust::device_vector<thrust::complex<FloatType>> kones;
  const thrust::complex<FloatType> * lnpsiGradientsPtr_dev_;
  thrust::device_vector<thrust::complex<FloatType>> avglnpsiGradients_dev_, z_dev_;
  thrust::device_vector<FloatType> diag_dev_;
  FloatType lambda_;
  cublasHandle_t theCublasHandle_;
};

namespace gpu_kernel
{
template <typename FloatType>
__global__ void SRCG__GetDiagOfSMatrix__(
  const thrust::complex<FloatType> * lnpsiGradients,
  const thrust::complex<FloatType> * avglnpsiGradients,
  const int nChains,
  const int nVariables,
  FloatType * diag
);

template <typename FloatType>
__global__ void SRCG__AddRegularization__(
  const FloatType lambda,
  const thrust::complex<FloatType> * a,
  const FloatType * diag,
  const int nVariables,
  thrust::complex<FloatType> * b
);

template <typename FloatType>
__global__ void SRCG__DiagonalPreconditioner__(
  const FloatType * diag,
  const thrust::complex<FloatType> * a,
  const int nVariables,
  const FloatType lambda,
  thrust::complex<FloatType> * b
);
} // namespace gpu_kernel


template <typename FloatType>
SMatrixForCG<FloatType>::SMatrixForCG(const int nChains, const int nVariables):
  knChains(nChains),
  knVariables(nVariables),
  kgpuBlockSize1(CHECK_BLOCK_SIZE(1+(nVariables-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize2(CHECK_BLOCK_SIZE(1+(nChains-1)/NUM_THREADS_PER_BLOCK)),
  kone(1, 0),
  kzero(0, 0),
  kmone(-1, 0),
  kinvNchains(static_cast<FloatType>(1)/nChains, static_cast<FloatType>(0)),
  kones(nChains, 1),
  avglnpsiGradients_dev_(nVariables, 0),
  diag_dev_(nVariables, 0),
  z_dev_(nChains),
  lambda_(0)
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_));
}

template <typename FloatType>
SMatrixForCG<FloatType>::~SMatrixForCG()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename FloatType>
void SMatrixForCG<FloatType>::set_lnpsiGradients(const thrust::device_vector<thrust::complex<FloatType>> & lnpsiGradients_dev, const FloatType lambda)
{
  if (knChains*knVariables != lnpsiGradients_dev.size())
    throw std::length_error("nChains*nVariables != lnpsiGradients_dev.size()");
  lnpsiGradientsPtr_dev_ = PTR_FROM_THRUST(lnpsiGradients_dev.data());
  lambda_ = lambda;
  // avglnpsiGradients_{i} = 1/nChains*\sum_{k}(lnpsiGradients_{k,i})
  cublas::gemv(theCublasHandle_, knVariables, knChains, kinvNchains, lnpsiGradientsPtr_dev_,
    PTR_FROM_THRUST(kones.data()), kzero, PTR_FROM_THRUST(avglnpsiGradients_dev_.data()));
  // diag_{i} = 1/nChains*\sum_{k}(|lnpsiGradients_{k,i}|^2) - |avglnpsiGradients_{i}|^2
  gpu_kernel::SRCG__GetDiagOfSMatrix__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(lnpsiGradientsPtr_dev_, PTR_FROM_THRUST(avglnpsiGradients_dev_.data()),
    knChains, knVariables, PTR_FROM_THRUST(diag_dev_.data()));
}

template <typename FloatType>
void SMatrixForCG<FloatType>::dot(const thrust::device_vector<thrust::complex<FloatType>> & a, thrust::device_vector<thrust::complex<FloatType>> & b)
{
  // z_{k} = \sum_{i}(lnpsiGradients_{k,i}*a_{i})
  cublas::gemm(theCublasHandle_, 1, knChains, knVariables, kone, kzero, PTR_FROM_THRUST(a.data()),
    lnpsiGradientsPtr_dev_, PTR_FROM_THRUST(z_dev_.data()));
  // z = conj(z)
  gpu_kernel::common__ApplyComplexConjugateVector__<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(z_dev_.data()), z_dev_.size());
  // tmp = conj(\sum_{i}(avglnpsiGradients_{i}*a_{i}))
  const thrust::complex<FloatType> tmp = thrust::conj(thrust::inner_product(thrust::device, avglnpsiGradients_dev_.begin(),
    avglnpsiGradients_dev_.end(), a.begin(), kzero));
  // b_{i} = avglnpsiGradients_{i}*tmp => \sum_{j}(avglnpsiGradients_{i}*conj(avglnpsiGradients_{j})*conj(a_{j}))
  thrust::transform(avglnpsiGradients_dev_.begin(), avglnpsiGradients_dev_.end(), b.begin(), b.begin(),
    internal_impl::AxpyFunctor<FloatType>(tmp, kzero));
  // b_{i} = 1/nChains*\sum_{k}(lnpsiGradients_{k,i}*z_{k}) - b_{i} 
  cublas::gemv(theCublasHandle_, knVariables, knChains, kinvNchains, lnpsiGradientsPtr_dev_, PTR_FROM_THRUST(z_dev_.data()),
    kmone, PTR_FROM_THRUST(b.data()));
  // conj(b) => (<var*var>-<var>*<var>)*a
  gpu_kernel::common__ApplyComplexConjugateVector__<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(b.data()), b.size());
  // b_{i} = lambda*a_{i}*diag_{i}+b_{i}
  gpu_kernel::SRCG__AddRegularization__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(lambda_, PTR_FROM_THRUST(a.data()),
    PTR_FROM_THRUST(diag_dev_.data()), knVariables, PTR_FROM_THRUST(b.data()));
}

template <typename FloatType>
void SMatrixForCG<FloatType>::applyPrecond(const thrust::device_vector<thrust::complex<FloatType>> & rhs_dev,
  thrust::device_vector<thrust::complex<FloatType>> & x_dev) const
{
  // Diagonal preconditioner
  gpu_kernel::SRCG__DiagonalPreconditioner__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(PTR_FROM_THRUST(diag_dev_.data()),
    PTR_FROM_THRUST(rhs_dev.data()), knVariables, lambda_, PTR_FROM_THRUST(x_dev.data()));
}

namespace gpu_kernel
{
template <typename FloatType>
__global__ void SRCG__GetDiagOfSMatrix__(
  const thrust::complex<FloatType> * lnpsiGradients,
  const thrust::complex<FloatType> * avglnpsiGradients,
  const int nChains,
  const int nVariables,
  FloatType * diag)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType invNsamples = 1/static_cast<FloatType>(nChains), zero = 0;
  while (idx < nVariables)
  {
    diag[idx] = zero;
    for (int k=0; k<nChains; ++k)
      diag[idx] += thrust::norm(lnpsiGradients[k*nVariables+idx]);
    diag[idx] = invNsamples*diag[idx]-thrust::norm(avglnpsiGradients[idx]);
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void SRCG__AddRegularization__(
  const FloatType lambda,
  const thrust::complex<FloatType> * a,
  const FloatType * diag,
  const int nVariables,
  thrust::complex<FloatType> * b)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nVariables)
  {
    b[idx] += lambda*diag[idx]*a[idx];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void SRCG__DiagonalPreconditioner__(
  const FloatType * diag,
  const thrust::complex<FloatType> * rhs,
  const int nVariables,
  const FloatType lambda,
  thrust::complex<FloatType> * x)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType one = 1;
  while (idx < nVariables)
  {
    x[idx] = rhs[idx]/((one+lambda)*diag[idx]);
    idx += nstep;
  }
}
} // namespace gpu_kernel
