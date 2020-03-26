// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iomanip>
#include <numeric>
#include "mcmc_sampler.hpp"
#include "blas_lapack.hpp"
#include "linear_solver.hpp"

// Ref. S. Sorella, M. Casula, and D. Rocca, J. Chem. Phys. 127, 014105 (2007).
template <typename FloatType, template<typename> class LinearSolver>
class StochasticReconfiguration
{
public:
  StochasticReconfiguration(const int nChains, const int nVariables);
  template <typename SamplerType>
  void propagate(SamplerType & sampler, const int nIteration, const int naccumulation,
    const int nMCSteps = 1, const FloatType deltaTau = 1e-3)
  {
    const std::complex<FloatType> oneOverTotalMeas = 1/static_cast<FloatType>(knChains*naccumulation);
    std::vector<std::complex<FloatType> > conjHavgArr(naccumulation, kzero);
    std::cout << "# of loop\t" << "<H>" << std::endl << std::setprecision(7);
    for (int n=0; n<nIteration; ++n)
    {
      std::cout << std::setw(5) << (n+1) << std::setw(16);
      // aO_i = (\sum_k lnpsigradients_ki)*oneOverTotalMeas
      std::fill(aO_.begin(), aO_.end(), kzero);
      // S_ij = (\sum_k (lnpsiGradients_ki)^H * lnpsiGradients_kj)*oneOverTotalMeas - (aO_i)^+ * aO_j
      std::fill(S_.begin(), S_.end(), kzero);
      // F_i = (\sum_k std::conj(htilda_k) * lnpsiGradients_ki)*oneOverTotalMeas - conjHavg*aO_i
      std::fill(F_.begin(), F_.end(), kzero);
      for (int nacc=0; nacc<naccumulation; ++nacc)
      {
        sampler.do_mcmc_steps(nMCSteps);
        sampler.get_htilda(&htilda_[0]);
        sampler.get_lnpsiGradients(&lnpsiGradients_[0]);
        // (1) aO_i = (\sum_k lnpsigradients_ki)*oneOverTotalMeas
        blas::gemv(knVariables, knChains, oneOverTotalMeas, &lnpsiGradients_[0], &kones[0], kone, &aO_[0]);
        // (1) S_ij = (\sum_k (lnpsiGradients_ki)^H * lnpsiGradients_kj)*oneOverTotalMeas
        blas::herk(knVariables, knChains, oneOverTotalMeas, &lnpsiGradients_[0], kone, &S_[0]);
        // (1) F_i = (\sum_k std::conj(htilda_k) * lnpsiGradients_ki)*oneOverTotalMeas
        for (int k=0; k<knChains; ++k)
          htilda_[k] = std::conj(htilda_[k]);
        blas::gemv(knVariables, knChains, oneOverTotalMeas, &lnpsiGradients_[0], &htilda_[0], kone, &F_[0]);
        conjHavgArr[nacc] = oneOverTotalMeas.real()*std::accumulate(htilda_.begin(), htilda_.end(), kzero);
      }
      // (2) S_ij -= (aO_i)^+ * aO_j (Note that 'uplo' is set to L!)
      blas::her(knVariables, kminusOne, &aO_[0], &S_[0]);
      const std::complex<FloatType> conjHavg = std::accumulate(conjHavgArr.begin(), conjHavgArr.end(), kzero);
      // (2) F = (F-conjHavg*aO_i)^+
      for (int i=0; i<knVariables; ++i)
        F_[i] = std::conj(F_[i]+kminusOne*conjHavg*aO_[i]);
      const FloatType lambda = this->schedular_();
      for (int i=0; i<knVariables; ++i)
      {
        // S_ij = (1 + lambda*\delta_ij)*S_ij
        S_[i*knVariables+i] = (1+lambda)*S_[i*knVariables+i];
        // transpose S_ to prepare as fortran style format
        for (int j=i+1; j<knVariables; ++j)
        {
          S_[j*knVariables+i] = S_[i*knVariables+j];
          S_[i*knVariables+j] = std::conj(S_[i*knVariables+j]);
        }
      }
      // F_ = S_^{-1}*F_
      linSolver_.solve(&S_[0], &F_[0]);
      sampler.evolve(&F_[0], deltaTau);
      std::cout << (conjHavg.real()) << std::endl << std::flush;
    }
  }
private:
  FloatType schedular_();
  std::vector<std::complex<FloatType> > htilda_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > aO_, S_, F_;
  const std::vector<std::complex<FloatType> > kones;
  const std::complex<FloatType> kone, kzero, kminusOne;
  const int knChains, knVariables;
  static constexpr FloatType klambda0 = 100.0, kb = 0.9, klambMin = 1e-2;
  int nIteration_;
  FloatType bp_;
  LinearSolver<FloatType> linSolver_;
};

#include "impl_optimizer.hpp"
