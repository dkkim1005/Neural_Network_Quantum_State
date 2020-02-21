// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <vector>
#include <complex>
#include <trng/yarn5.hpp>
#include <trng/uniform01_dist.hpp>
#include <trng/uniform_int_dist.hpp>

/*
 * Base class of importance sampling for wave functions: ln(\psi(x))
 *  - ratio = norm(ln(\psi(x1))-ln(\psi(x0)))
 *   where x1 is a candidate of the next state and x0 is a current state.
 */
template <template<typename> class DerivedParallelSampler, typename Properties>
class BaseParallelSampler
{
public:
  BaseParallelSampler(const int nSites, const int nChains,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void warm_up(const int nMCSteps = 100);
  void do_mcmc_steps(const int nMCSteps = 1);
  void get_htilda(std::complex<typename Properties::FloatType> * htilda);
  void get_lnpsiGradients(std::complex<typename Properties::FloatType> * lnpsiGradients);
  int get_nChains() const { return knChains; }
  void evolve(const std::complex<typename Properties::FloatType> * trueGradients,
    const typename Properties::FloatType learningRate);
private:
  const int knMCUnitSteps, knChains;
  std::vector<bool> updateList_;
  std::vector<typename Properties::FloatType> ratio_;
  std::vector<trng::yarn5> randDev_;
  trng::uniform01_dist<typename Properties::FloatType> randUniform_;
protected:
  std::vector<std::complex<typename Properties::FloatType> > lnpsi1_, lnpsi0_;
};

// macro expression to be used at derived classes
#define USING_OF_BASE_PARALLEL_SAMPLER(DERIVED_PARALLEL_SAMPLER, PROPERTIES)\
friend BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, PROPERTIES>;\
using BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, PROPERTIES>::lnpsi1_;\
using BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, PROPERTIES>::lnpsi0_;

#include "impl_mcmc_sampler.hpp"
