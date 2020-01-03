#pragma once

#include "blas.hpp"
#include "hamiltonians.hpp"
#include <numeric>
#include "linear_solver.hpp"

template <typename float_t>
class StochasticReconfiguration
{
public:
  StochasticReconfiguration(const int nChains, const int nVariables);
  template <typename WFSampler>
  void propagate(BaseParallelVMC<WFSampler, float_t> & sampler, const int nIteration, const int nMCSteps = 1, const float_t deltaTau = 1e-4)
  {
    for (int n=0; n<nIteration; ++n)
    {
      //std::cout << "n: " << (n+1) << " ";
      sampler.do_mcmc_steps(nMCSteps);
      sampler.get_htilda(&htilda_[0]);
      sampler.get_lnpsiGradients(&lnpsiGradients_[0]);
      // aO_i = (\sum_k lnpsigradients_ki)/knChains
      blas::gemv(knVariables, knChains, koneOverNchains, &lnpsiGradients_[0], &kones[0], kzero, &aO_[0]);
      // aOO_ij = -(aO_i)^+ * aO_j (Note that 'uplo' is set to L!)
      std::fill(aOO_.begin(), aOO_.end(), kzero);
      blas::her(knVariables, kminusOne, &aO_[0], &aOO_[0]);
      // aOO_ij += (\sum_k (lnpsiGradients_ki)^H * lnpsiGradients_kj)/knChains
      blas::herk(knVariables, knChains, koneOverNchains, &lnpsiGradients_[0], kone, &aOO_[0]);
      // aOO^{reg}_ij = aOO_ij + lambda*\delta_ij*aOO_ij
      const float_t lambda = this->schedular();
      for (int i=0; i<knVariables; ++i)
        aOO_[i*knVariables+i] = (1+lambda)*aOO_[i*knVariables+i];
      // F_i = -(\frac{1}{knChains}\sum_k std::conj(htilda_k))*aO_i
      for (int k=0; k<knChains; ++k)
        htilda_[k] = std::conj(htilda_[k]);
      std::complex<float_t> conjHavg = koneOverNchains*std::accumulate(htilda_.begin(), htilda_.end(), kzero);
      for (int i=0; i<knVariables; ++i)
        F_[i] = kminusOne*conjHavg*aO_[i];
      // F_i += \frac{1}{knChains}\sum_k std::conj(htilda_k) * lnpsiGradients_ki
      blas::gemv(knVariables, knChains, koneOverNchains, &lnpsiGradients_[0], &htilda_[0], kone, &F_[0]);
      for (int i=0; i<knVariables; ++i)
        F_[i] = std::conj(F_[i]);
      // F_ = aOO_^{-1}*F_
      linSolver_.solve(&aOO_[0], &F_[0], 'L');
      sampler.evolve(&F_[0], deltaTau);
      //std::cout << (conjHavg.real()) << std::endl;
    }
  }
private:
  float_t schedular();
  std::vector<std::complex<float_t> > htilda_, lnpsiGradients_;
  std::vector<std::complex<float_t> > aOO_, aO_, F_;
  const std::vector<std::complex<float_t> > kones;
  const std::complex<float_t> koneOverNchains, kone, kzero, kminusOne;
  const int knChains, knVariables;
  static constexpr float_t klambda0 = 100.0, kb = 0.9, klambMin = 5e-2;
  int nIteration_;
  float_t kbp_;
  LinearSolver<float_t> linSolver_;
};

template <typename float_t>
StochasticReconfiguration<float_t>::StochasticReconfiguration(const int nChains, const int nVariables):
htilda_(nChains),
lnpsiGradients_(nChains*nVariables),
kones(nChains, std::complex<float_t>(1.0, 0.0)),
koneOverNchains(std::complex<float_t>(1.0/static_cast<float_t>(nChains), 0.0)),
kone(std::complex<float_t>(1.0, 0.0)),
kzero(std::complex<float_t>(0.0, 0.0)),
kminusOne(std::complex<float_t>(-1.0, 0.0)),
knChains(nChains),
knVariables(nVariables),
aOO_(nVariables*nVariables),
aO_(nVariables),
F_(nVariables),
nIteration_(0),
kbp_(1.0),
linSolver_(nVariables)
{}

template <typename float_t>
float_t StochasticReconfiguration<float_t>::schedular()
{
  kbp_ *= kb;
  const float_t lambda = klambda0*kbp_;
  return ((lambda > klambMin) ? lambda : klambMin);
}
