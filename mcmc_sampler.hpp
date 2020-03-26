// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <vector>
#include <complex>
#include <trng/yarn2.hpp>
#include <trng/yarn5.hpp>
#include <trng/yarn5s.hpp>
#include <trng/uniform01_dist.hpp>
#include <trng/uniform_int_dist.hpp>

/*
 * Base class of importance sampling for wave functions: ln(\psi(x))
 *  - ratio = norm(ln(\psi(x1))-ln(\psi(x0)))
 *   where x1 is a candidate of the next state and x0 is a current state.
 */
template <template<typename> class DerivedParallelSampler, typename TraitsClass>
class BaseParallelSampler
{
  using RandEngineType = trng::yarn2;
  using FloatType = typename TraitsClass::FloatType;
public:
  BaseParallelSampler(const int nSites, const int nChains,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void warm_up(const int nMCSteps = 100);
  void do_mcmc_steps(const int nMCSteps = 1);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  int get_nChains() const { return knChains; }
  void evolve(const std::complex<FloatType> * trueGradients,
    const FloatType learningRate);
private:
  const int knMCUnitSteps, knChains;
  std::vector<bool> updateList_;
  std::vector<FloatType> ratio_;
  std::vector<RandEngineType> randDev_;
  trng::uniform01_dist<FloatType> randUniform_;
protected:
  std::vector<std::complex<FloatType> > lnpsi1_, lnpsi0_;
};

// macro expression to be used at derived classes
#define USING_OF_BASE_PARALLEL_SAMPLER(DERIVED_PARALLEL_SAMPLER, PROPERTIES)\
friend BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, PROPERTIES>;\
using BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, PROPERTIES>::lnpsi1_;\
using BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, PROPERTIES>::lnpsi0_;


template <template<typename> class DerivedParallelTemperingSampler, typename TraitsClass>
class BaseParallelTemperingSampler
{
  using RandEngineType = trng::yarn5s;
  using FloatType = typename TraitsClass::FloatType;
public:
  BaseParallelTemperingSampler(const int nSites, const int nChainsPerBeta, const int nBeta,
    const unsigned long seedDistance, const unsigned long seedNumber = 0);
  void warm_up(const int nMCSteps = 100);
  void do_mcmc_steps(const int nMCSteps);
  void get_htilda(std::complex<FloatType> * htilda);
  void get_lnpsiGradients(std::complex<FloatType> * lnpsiGradients);
  void evolve(const std::complex<FloatType> * trueGradients, const FloatType learningRate);
private:
  const int knMCUnitSteps, knTotChains, knChainsPerBeta, knBeta;
  std::vector<bool> updateList_;
  std::vector<FloatType> ratio_, beta_;
  std::vector<RandEngineType> randDev_;
  trng::uniform01_dist<FloatType> randUniform_;
protected:
  std::vector<std::complex<FloatType> > lnpsi1_, lnpsi0_;
};

// macro expression to be used at derived classes
#define USING_OF_BASE_PARALLEL_TEMPERING_SAMPLER(DERIVED_SAMPLER, PROPERTIES)\
friend BaseParallelTemperingSampler<DERIVED_SAMPLER, PROPERTIES>;\
using BaseParallelTemperingSampler<DERIVED_SAMPLER, PROPERTIES>::lnpsi1_;\
using BaseParallelTemperingSampler<DERIVED_SAMPLER, PROPERTIES>::lnpsi0_;

#include "impl_mcmc_sampler.hpp"
