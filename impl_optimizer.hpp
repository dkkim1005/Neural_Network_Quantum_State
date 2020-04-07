// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <typename FloatType, template<typename> class LinearSolver>
StochasticReconfiguration<FloatType, LinearSolver>::StochasticReconfiguration(const int nChains, const int nVariables):
  htilda_(nChains),
  lnpsiGradients_(nChains*nVariables),
  kones(nChains, std::complex<FloatType>(1.0, 0.0)),
  kone(std::complex<FloatType>(1.0, 0.0)),
  kzero(std::complex<FloatType>(0.0, 0.0)),
  kminusOne(std::complex<FloatType>(-1.0, 0.0)),
  knChains(nChains),
  knVariables(nVariables),
  S_(nVariables*nVariables),
  aO_(nVariables),
  F_(nVariables),
  nIteration_(0),
  bp_(1.0),
  linSolver_(nVariables) {}

template <typename FloatType, template<typename> class LinearSolver>
FloatType StochasticReconfiguration<FloatType, LinearSolver>::schedular_()
{
  bp_ *= kb;
  const FloatType lambda = klambda0*bp_;
  return ((lambda > klambMin) ? lambda : klambMin);
}


template <typename FloatType>
StochasticGradientDescent<FloatType>::StochasticGradientDescent(const int nChains, const int nVariables):
  htilda_(nChains),
  lnpsiGradients_(nChains*nVariables),
  kones(nChains, std::complex<FloatType>(1.0, 0.0)),
  kone(std::complex<FloatType>(1.0, 0.0)),
  kzero(std::complex<FloatType>(0.0, 0.0)),
  kminusOne(std::complex<FloatType>(-1.0, 0.0)),
  knChains(nChains),
  knVariables(nVariables),
  S_(nVariables),
  aO_(nVariables),
  F_(nVariables),
  nIteration_(0),
  bp_(1.0) {}

template <typename FloatType>
FloatType StochasticGradientDescent<FloatType>::schedular_()
{
  bp_ *= kb;
  const FloatType lambda = klambda0*bp_;
  return ((lambda > klambMin) ? lambda : klambMin);
}
