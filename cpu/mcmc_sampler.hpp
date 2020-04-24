// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <numeric>
#include <random>
#include <exception>
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
  FloatType meas_acceptance_ratio();
private:
  const int knMCUnitSteps, knChains;
  unsigned long totalMeasurements_;
  std::vector<bool> updateList_;
  std::vector<FloatType> ratio_;
  std::vector<RandEngineType> randDev_;
  std::vector<unsigned long> acceptanceRatio_;
  trng::uniform01_dist<FloatType> randUniform_;
protected:
  std::vector<std::complex<FloatType> > lnpsi1_, lnpsi0_;
};

// macro expression to be used at derived classes
#define USING_OF_BASE_PARALLEL_SAMPLER(DERIVED_PARALLEL_SAMPLER, PROPERTIES)\
friend BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, PROPERTIES>;\
using BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, PROPERTIES>::lnpsi1_;\
using BaseParallelSampler<DERIVED_PARALLEL_SAMPLER, PROPERTIES>::lnpsi0_


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
  FloatType meas_acceptance_ratio();
private:
  const int knMCUnitSteps, knTotChains, knChainsPerBeta, knBeta;
  unsigned long totalMeasurements_;
  std::vector<bool> updateList_;
  std::vector<FloatType> ratio_, beta_;
  std::vector<unsigned long> acceptanceRatio_;
  std::vector<RandEngineType> randDev_;
  trng::uniform01_dist<FloatType> randUniform_;
protected:
  std::vector<std::complex<FloatType> > lnpsi1_, lnpsi0_;
};

// macro expression to be used at derived classes
#define USING_OF_BASE_PARALLEL_TEMPERING_SAMPLER(DERIVED_SAMPLER, PROPERTIES)\
friend BaseParallelTemperingSampler<DERIVED_SAMPLER, PROPERTIES>;\
using BaseParallelTemperingSampler<DERIVED_SAMPLER, PROPERTIES>::lnpsi1_;\
using BaseParallelTemperingSampler<DERIVED_SAMPLER, PROPERTIES>::lnpsi0_

/*
 * return a randomly shuffled array: [0,1,2,3,4,5,...] => [[4,0,1,...],[2,3,5,...],[...],...]
 */
template <typename RandEngineType>
class RandomBatchIndexing
{
public:
  RandomBatchIndexing(const int size, const double rate);
  const std::vector<int> & get_miniBatch() const;
  void next();
private:
  std::vector<int> indexOfFullBatch_;
  std::vector<std::vector<int> > indexOfMiniBatch_;
  int batchSetIdx_;
  RandEngineType rng_;
};

#include "impl_mcmc_sampler.hpp"
