// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <omp.h>
#include <vector>
#include <complex>
#include <chrono>
#include <cstring>
#include <iomanip>
#include "blas_lapack.hpp"
#include "common.hpp"

namespace spinhalfsystem
{
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
  void save(const RBMDataType typeInfo, const std::string filePath,
    const int precision = FloatTypeTrait_<FloatType>::precision) const;
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


/*
 * W1: weight matrix positioned between the input and hidden layers
 * W2: weight matrix  "" the hidden and output layers
 * B1: bias vector added to the hidden layer
 */
enum class FNNDataType { W1, W2, B1 };

template <typename FloatType>
class ComplexFNN
{
public:
  ComplexFNN(const int nInputs, const int nHiddens, const int nChains);
  ComplexFNN(const ComplexFNN & rhs) = delete;
  ComplexFNN & operator=(const ComplexFNN & rhs) = delete;
  void update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate);
  void initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates = NULL);
  void forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi);
  void backward(std::complex<FloatType> * lnpsiGradients);
  void load(const FNNDataType typeInfo, const std::string filePath);
  void save(const FNNDataType typeInfo, const std::string filePath,
    const int precision = FloatTypeTrait_<FloatType>::precision) const;
  void spin_flip(const std::vector<bool> & doSpinFlip);
  const std::complex<FloatType> * get_spinStates() const { return &spinStates_[0]; };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nVariables() const { return variables_.size(); }
private:
  const int knInputs, knHiddens, knChains;
  const std::complex<FloatType> kzero, kone, ktwo;
  const std::vector<std::complex<FloatType> > koneChains;
  std::vector<std::complex<FloatType> > variables_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > spinStates_;
  std::vector<std::complex<FloatType> > y_, acty_;
  std::complex<FloatType> * wi1_, * w1o_, * b1_, * d_dwi1_, * d_dw1o_, * d_db1_;
  int index_;
};
} // namespace spinhalfsystem
#include "impl_neural_quantum_state.hpp"
