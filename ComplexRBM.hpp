// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <omp.h>
#include <vector>
#include <complex>
#include <chrono>
#include <cstring>
#include <iomanip>
#include "blas_lapack.hpp"

template <typename FloatType> struct FloatTypeTrait_ {};
template <> struct FloatTypeTrait_<float> { static constexpr int precision = 8; };
template <> struct FloatTypeTrait_<double> { static constexpr int precision = 15; };

/*
 * W : weight matrix
 * V : vector for visible units
 * H : vector for hidden units
 */
enum class RBMDataType { W, V, H };

template <typename FloatType>
class ComplexRBM
{
public:
  ComplexRBM(const int nInputs, const int nHiddens, const int nChains);
  ComplexRBM(const ComplexRBM & rhs) = delete;
  ComplexRBM & operator=(const ComplexRBM & rhs) = delete;
  void update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate);
  void initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates = NULL);
  void forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi);
  void backward(std::complex<FloatType> * lnpsiGradients);
  void load(const RBMDataType typeInfo, const std::string filePath);
  void save(const RBMDataType typeInfo, const std::string filePath, const int precision = FloatTypeTrait_<FloatType>::precision) const;
  void spin_flip(const std::vector<bool> & doSpinFlip);
  const std::complex<FloatType> * get_spinStates() const { return &spinStates_[0]; };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nVariables() const { return variables_.size(); }
private:
  const int knInputs, knHiddens, knChains;
  const std::vector<std::complex<FloatType> > koneChains, koneHiddens;
  const std::complex<FloatType> kzero, kone, ktwo;
  std::vector<std::complex<FloatType> > variables_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > spinStates_;
  std::vector<std::complex<FloatType> > y_, ly_, sa_;
  std::complex<FloatType> * w_, * a_, * b_, * d_dw_, * d_da_, * d_db_;
  int index_;
};

#include "impl_ComplexRBM.hpp"
