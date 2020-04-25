// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iostream>
#include <iomanip>
#include "common.cuh"
#include "cublas_template.cuh"
#include "linear_solver.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

namespace gpu_kernel
{
template <typename FloatType>
__global__ void SR__FStep2__(
  const thrust::complex<FloatType> conjHavg,
  const thrust::complex<FloatType> * aO,
  const int nVariables,
  thrust::complex<FloatType> * F
);

template <typename FloatType>
__global__ void SR__ArrangeSmatrix__(
  const FloatType lambda,
  const int nVariables,
  thrust::complex<FloatType> * S
);
} // namespace gpu_kernel

// Ref. S. Sorella, M. Casula, and D. Rocca, J. Chem. Phys. 127, 014105 (2007).
template <typename FloatType, template<typename> class LinearSolver>
class StochasticReconfiguration
{
public:
  StochasticReconfiguration(const int nChains, const int nVariables);
  ~StochasticReconfiguration();
  template <template<typename> class SamplerType, typename Traits>
  void propagate(BaseParallelSampler<SamplerType, Traits> & sampler, const int nIteration, const int naccumulation,
    const int nMCSteps = 1, const FloatType deltaTau = 1e-3)
  {
    const thrust::complex<FloatType> oneOverTotalMeas = 1/static_cast<FloatType>(knChains*naccumulation);
    thrust::host_vector<thrust::complex<FloatType>> conjHavgArr_host(naccumulation, kzero);
    std::cout << "# of loop\t" << "<H>" << std::endl << std::setprecision(7);
    for (int n=0; n<nIteration; ++n)
    {
      std::cout << std::setw(5) << (n+1) << std::setw(16);
      // aO_i = (\sum_k lnpsigradients_ki)*oneOverTotalMeas
      thrust::fill(aO_dev_.begin(), aO_dev_.end(), kzero);
      // S_ij = (\sum_k (lnpsiGradients_ki)^H * lnpsiGradients_kj)*oneOverTotalMeas - (aO_i)^+ * aO_j
      thrust::fill(S_dev_.begin(), S_dev_.end(), kzero);
      // F_i = (\sum_k std::conj(htilda_k) * lnpsiGradients_ki)*oneOverTotalMeas - conjHavg*aO_i
      thrust::fill(F_dev_.begin(), F_dev_.end(), kzero);
      for (int nacc=0; nacc<naccumulation; ++nacc)
      {
        sampler.do_mcmc_steps(nMCSteps);
        sampler.get_htilda(PTR_FROM_THRUST(htilda_dev_.data()));
        sampler.get_lnpsiGradients(PTR_FROM_THRUST(lnpsiGradients_dev_.data()));
        // (1) aO_i = (\sum_k lnpsigradients_ki)*oneOverTotalMeas
        cublas::gemv(theCublasHandle_, knVariables, knChains, oneOverTotalMeas, PTR_FROM_THRUST(lnpsiGradients_dev_.data()),
          PTR_FROM_THRUST(kones_dev.data()), kone, PTR_FROM_THRUST(aO_dev_.data()));
        // (1) S_ij = (\sum_k (lnpsiGradients_ki)^H * lnpsiGradients_kj)*oneOverTotalMeas
        cublas::herk(theCublasHandle_, knVariables, knChains, oneOverTotalMeas.real(), PTR_FROM_THRUST(lnpsiGradients_dev_.data()),
          kone.real(), PTR_FROM_THRUST(S_dev_.data()));
        // (1) F_i = (\sum_k std::conj(htilda_k) * lnpsiGradients_ki)*oneOverTotalMeas
        gpu_kernel::common__ApplyComplexConjugateVector__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>
          (PTR_FROM_THRUST(htilda_dev_.data()), htilda_dev_.size());
        cublas::gemv(theCublasHandle_, knVariables, knChains, oneOverTotalMeas, PTR_FROM_THRUST(lnpsiGradients_dev_.data()),
          PTR_FROM_THRUST(htilda_dev_.data()), kone, PTR_FROM_THRUST(F_dev_.data()));
        conjHavgArr_host[nacc] = oneOverTotalMeas.real()*thrust::reduce(thrust::device, htilda_dev_.begin(), htilda_dev_.end(), kzero);
      }
      // (2) S_ij -= (aO_i)^+ * aO_j (Note that 'uplo' is set to L!)
      cublas::her(theCublasHandle_, knVariables, kminusOne.real(), PTR_FROM_THRUST(aO_dev_.data()), PTR_FROM_THRUST(S_dev_.data()));
      const thrust::complex<FloatType> conjHavg = thrust::reduce(thrust::host, conjHavgArr_host.begin(), conjHavgArr_host.end(), kzero);
      // (2) F = (F-conjHavg*aO_i)^+
      gpu_kernel::SR__FStep2__<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(conjHavg, PTR_FROM_THRUST(aO_dev_.data()),
        knVariables, PTR_FROM_THRUST(F_dev_.data()));
      const FloatType lambda = this->schedular_();
      gpu_kernel::SR__ArrangeSmatrix__<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(lambda, knVariables, PTR_FROM_THRUST(S_dev_.data()));
      // dx_ = S_^{-1}*F_
      linSolver_.solve(PTR_FROM_THRUST(S_dev_.data()), PTR_FROM_THRUST(F_dev_.data()), PTR_FROM_THRUST(dx_dev_.data()));
      sampler.evolve(PTR_FROM_THRUST(dx_dev_.data()), deltaTau);
      cudaDeviceSynchronize();
      std::cout << conjHavg.real() << std::endl << std::flush;
    }
  }
private:
  FloatType schedular_();
  thrust::device_vector<thrust::complex<FloatType>> htilda_dev_, lnpsiGradients_dev_;
  thrust::device_vector<thrust::complex<FloatType>> aO_dev_, S_dev_, F_dev_, dx_dev_;
  const thrust::device_vector<thrust::complex<FloatType>> kones_dev;
  const thrust::complex<FloatType> kone, kzero, kminusOne; 
  const int knChains, knVariables, kgpuBlockSize1, kgpuBlockSize2;
  static constexpr FloatType klambda0 = 100.0, kb = 0.9, klambMin = 1e-4;
  int nIteration_;
  FloatType bp_;
  LinearSolver<FloatType> linSolver_;
  cublasHandle_t theCublasHandle_;
};

#include "impl_optimizer.cuh"