// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <typename FloatType, template<typename> class LinearSolver>
StochasticReconfiguration<FloatType, LinearSolver>::StochasticReconfiguration(const int nChains, const int nVariables):
  htilda_dev_(nChains),
  lnpsiGradients_dev_(nChains*nVariables),
  kones_dev(nChains, thrust::complex<FloatType>(1.0, 0.0)),
  kone(thrust::complex<FloatType>(1.0, 0.0)),
  kzero(thrust::complex<FloatType>(0.0, 0.0)),
  kminusOne(thrust::complex<FloatType>(-1.0, 0.0)),
  knChains(nChains),
  knVariables(nVariables),
  S_dev_(nVariables*nVariables),
  aO_dev_(nVariables),
  F_dev_(nVariables),
  dx_dev_(nVariables),
  nIteration_(0),
  bp_(1.0),
  linSolver_(nVariables),
  kgpuBlockSize1(1+(nChains-1)/NUM_THREADS_PER_BLOCK),
  kgpuBlockSize2(1+(nVariables-1)/NUM_THREADS_PER_BLOCK)
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
}

template <typename FloatType, template<typename> class LinearSolver>
StochasticReconfiguration<FloatType, LinearSolver>::~StochasticReconfiguration()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename FloatType, template<typename> class LinearSolver>
FloatType StochasticReconfiguration<FloatType, LinearSolver>::schedular_()
{
  bp_ *= kb;
  const FloatType lambda = klambda0*bp_;
  return ((lambda > klambMin) ? lambda : klambMin);
}

namespace gpu_kernel
{
template <typename FloatType>
__global__ void SR__FStep2__(
  const thrust::complex<FloatType> conjHavg,
  const thrust::complex<FloatType> * aO,
  const int nVariables,
  thrust::complex<FloatType> * F)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nVariables)
  {
    F[idx] = thrust::conj(F[idx]-conjHavg*aO[idx]);
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void SR__ArrangeSmatrix__(
  const FloatType lambda,
  const int nVariables,
  thrust::complex<FloatType> * S)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType one = 1;
  // S_ij = (1+lambda*\delta_ij)*S_ij
  while (idx < nVariables)
  {
    //S[idx*nVariables+idx] = S[idx*nVariables+idx]+lambda;
    S[idx*nVariables+idx] = (one+lambda)*S[idx*nVariables+idx];
    // transpose S_ to set row-major order(fortran style format)
    for (int j=idx+1; j<nVariables; ++j)
    {
      S[j*nVariables+idx] = S[idx*nVariables+j];
      S[idx*nVariables+j] = thrust::conj(S[idx*nVariables+j]);
    }
    idx += nstep;
  }
}
} // namespace gpu_kernel
