// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iomanip>
#include <numeric>
#include <trng/yarn5.hpp>
#include <trng/uniform01_dist.hpp>
#include <trng/uniform_int_dist.hpp>
#include "blas_lapack.hpp"
#include "linear_solver.hpp"

/*
 * Base class of importance sampling for wave functions: ln(\psi(x))
 *  - ratio = norm(ln(\psi(x1))-ln(\psi(x0)))
 *   where x1 is a candidate of the next state and x0 is a current state.
 */
template <typename DerivedWFSampler, typename FloatType>
class BaseParallelVMC
{
public:
  BaseParallelVMC(const int nSites, const int nChains,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void warm_up(const int nMCSteps = 100);
  void do_mcmc_steps(const int nMCSteps = 1);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  int get_nChains() const { return knChains; }
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
private:
  const int knMCUnitSteps, knChains;
  std::vector<bool> updateList_;
  std::vector<FloatType> ratio_;
  std::vector<trng::yarn5> randDev_;
  trng::uniform01_dist<FloatType> randUniform_;
protected:
  std::vector<std::complex<FloatType> > lnpsi1_, lnpsi0_;
};

// Ref. S. Sorella, M. Casula, and D. Rocca, J. Chem. Phys. 127, 014105 (2007).
template <typename FloatType, template<typename> class LinearSolver>
class StochasticReconfiguration
{
public:
  StochasticReconfiguration(const int nChains, const int nVariables);
  template <typename WFSampler>
  void propagate(BaseParallelVMC<WFSampler, FloatType> & sampler,
    const int nIteration, const int nMCSteps = 1, const FloatType deltaTau = 1e-3)
  {
    std::cout << "# of loop\t" << "<H>" << std::endl << std::setprecision(7);
    for (int n=0; n<nIteration; ++n)
    {
      std::cout << std::setw(5) << (n+1) << std::setw(16);
      sampler.do_mcmc_steps(nMCSteps);
      sampler.get_htilda(&htilda_[0]);
      sampler.get_lnpsiGradients(&lnpsiGradients_[0]);
      // aO_i = (\sum_k lnpsigradients_ki)/knChains
      blas::gemv(knVariables, knChains, koneOverNchains, &lnpsiGradients_[0], &kones[0], kzero, &aO_[0]);
      // S_ij = -(aO_i)^+ * aO_j (Note that 'uplo' is set to L!)
      std::fill(S_.begin(), S_.end(), kzero);
      blas::her(knVariables, kminusOne, &aO_[0], &S_[0]);
      // S_ij += (\sum_k (lnpsiGradients_ki)^H * lnpsiGradients_kj)/knChains
      blas::herk(knVariables, knChains, koneOverNchains, &lnpsiGradients_[0], kone, &S_[0]);
      // aOO^{reg}_ij = S_ij + lambda*\delta_ij*S_ij
      const FloatType lambda = this->schedular();
      for (int i=0; i<knVariables; ++i)
        S_[i*knVariables+i] = (1+lambda)*S_[i*knVariables+i];
      // F_i = -(\frac{1}{knChains}\sum_k std::conj(htilda_k))*aO_i
      for (int k=0; k<knChains; ++k)
        htilda_[k] = std::conj(htilda_[k]);
      const std::complex<FloatType> conjHavg = koneOverNchains*std::accumulate(htilda_.begin(), htilda_.end(), kzero);
      for (int i=0; i<knVariables; ++i)
        F_[i] = kminusOne*conjHavg*aO_[i];
      // F_i += \frac{1}{knChains}\sum_k std::conj(htilda_k) * lnpsiGradients_ki
      blas::gemv(knVariables, knChains, koneOverNchains, &lnpsiGradients_[0], &htilda_[0], kone, &F_[0]);
      for (int i=0; i<knVariables; ++i)
        F_[i] = std::conj(F_[i]);
      // transpose S_ to prepare as fortran style format
      for (int i=0; i<knVariables; ++i)
        for (int j=i+1; j<knVariables; ++j)
        {
          S_[j*knVariables+i] = S_[i*knVariables+j];
          S_[i*knVariables+j] = std::conj(S_[i*knVariables+j]);
        }
      // F_ = S_^{-1}*F_
      linSolver_.solve(&S_[0], &F_[0]);
      sampler.evolve(&F_[0], deltaTau);
      std::cout << (conjHavg.real()) << std::endl << std::flush;
    }
  }
private:
  FloatType schedular();
  std::vector<std::complex<FloatType> > htilda_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > S_, aO_, F_;
  const std::vector<std::complex<FloatType> > kones;
  const std::complex<FloatType> koneOverNchains, kone, kzero, kminusOne;
  const int knChains, knVariables;
  static constexpr FloatType klambda0 = 100.0, kb = 0.9, klambMin = 1e-2;
  int nIteration_;
  FloatType bp_;
  LinearSolver<FloatType> linSolver_;
};

#include "impl_optimization.hpp"
