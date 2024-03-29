// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <omp.h>
#include <vector>
#include <array>
#include <complex>
#include <chrono>
#include <cstring>
#include <iomanip>
#include "blas_lapack.hpp"
#include "common.hpp"

namespace spinhalf
{
/*
 * W : weight matrix
 * V : vector for visible units
 * H : vector for hidden units
 */
enum class RBMDataType { W, V, H };

template <typename FloatType>
class RBM
{
public:
  RBM(const int nInputs, const int nHiddens, const int nChains);
  RBM(const RBM & rhs) = delete;
  RBM & operator=(const RBM & rhs) = delete;
  void update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate);
  void update_partial_variables(const std::complex<FloatType> * derivativeLoss,
    const FloatType learningRate, const std::vector<int> & hiddenNodes);
  void initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates = NULL);
  void forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi);
  void forward(const std::vector<int> & spinFlipIndex, std::complex<FloatType> * lnpsi);
  void forward(const std::vector<std::vector<int> > & spinFlipIndexes, std::complex<FloatType> * lnpsi);
  void backward(std::complex<FloatType> * lnpsiGradients);
  void partial_backward(std::complex<FloatType> * lnpsiGradients, const int & nChains);
  void partial_backward(std::complex<FloatType> * lnpsiGradients, const std::vector<int> & hiddenNodes);
  void load(const RBMDataType typeInfo, const std::string filePath);
  void save(const RBMDataType typeInfo, const std::string filePath,
    const int precision = FloatTypeTrait_<FloatType>::precision) const;
  void spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex = -1);
  void spin_flip(const std::vector<bool> & doSpinFlip, const std::vector<std::vector<int> > & spinFlipIndexes);
  void swap_states(const int & k1, const int & k2);
  const std::complex<FloatType> * get_spinStates() const { return &spinStates_[0]; };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nHiddens() const { return knHiddens; }
  int get_nVariables() const { return variables_.size(); }
private:
  const int knInputs, knHiddens, knChains;
  const std::vector<std::complex<FloatType> > koneChains, koneHiddens;
  const std::array<std::complex<FloatType>, 2> ktwoTrueFalse;
  const std::complex<FloatType> kzero, kone;
  const FloatType ktwo;
  std::vector<std::complex<FloatType> > variables_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > spinStates_;
  std::vector<std::complex<FloatType> > y_, ly_, sa_;
  std::complex<FloatType> * w_, * a_, * b_, * d_dw_, * d_da_, * d_db_;
  int index_;
};


// complex RBM with spatial translational symmetry (periodic boundary condition is emposed.)
template <typename FloatType>
class RBMTrSymm
{
public:
  RBMTrSymm(const int nInputs, const int alpha, const int nChains);
  RBMTrSymm(const RBMTrSymm & rhs) = delete;
  RBMTrSymm & operator=(const RBMTrSymm & rhs) = delete;
  void update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate);
  void initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates = NULL);
  void forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi);
  void backward(std::complex<FloatType> * lnpsiGradients);
  void load(const std::string filePath);
  void save(const std::string filePath, const int precision = FloatTypeTrait_<FloatType>::precision) const;
  void spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex = -1);
  const std::complex<FloatType> * get_spinStates() const { return &spinStates_[0]; };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nHiddens() const { return kAlpha*knInputs; }
  int get_nVariables() const { return variables_.size(); }
private:
  void construct_weight_and_bias_();

  const int knInputs, kAlpha, knChains;
  const std::vector<std::complex<FloatType> > koneChains, koneHiddens;
  const std::array<std::complex<FloatType>, 2> ktwoTrueFalse;
  const std::complex<FloatType> kzero, kone;
  const FloatType ktwo;
  std::vector<std::complex<FloatType> > variables_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > spinStates_;
  std::vector<std::complex<FloatType> > y_, ly_, sa_;
  std::vector<std::complex<FloatType> > wf_, bf_, af_;
  std::complex<FloatType> * w_, * a_, * b_, * d_dw_, * d_da_, * d_db_;
  int index_;
};


template <typename FloatType>
class RBMSfSymm
{
public:
  RBMSfSymm(const int nInputs, const int alpha, const int nChains);
  RBMSfSymm(const RBMSfSymm & rhs) = delete;
  RBMSfSymm & operator=(const RBMSfSymm & rhs) = delete;
  void update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate);
  void initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates = NULL);
  void forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi);
  void backward(std::complex<FloatType> * lnpsiGradients);
  void load(const std::string filePath);
  void save(const std::string filePath, const int precision = FloatTypeTrait_<FloatType>::precision) const;
  void spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex = -1);
  const std::complex<FloatType> * get_spinStates() const { return &spinStates_[0]; };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nVariables() const { return variables_.size(); }
private:
  const int knInputs, kAlpha, knChains;
  const std::vector<std::complex<FloatType> > koneChains, koneHiddens;
  const std::array<std::complex<FloatType>, 2> ktwoTrueFalse;
  const std::complex<FloatType> kzero, kone;
  const FloatType ktwo;
  std::vector<std::complex<FloatType> > variables_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > spinStates_;
  std::vector<std::complex<FloatType> > y_, ly_;
  std::complex<FloatType> * w_, * d_dw_;
  int index_;
};


/*
 * W1: weight matrix positioned between the input and hidden layers
 * W2: weight matrix  "" the hidden and output layers
 * B1: bias vector added to the hidden layer
 */
enum class FFNNDataType { W1, W2, B1 };

template <typename FloatType>
class FFNN
{
public:
  FFNN(const int nInputs, const int nHiddens, const int nChains);
  FFNN(const FFNN & rhs) = delete;
  FFNN & operator=(const FFNN & rhs) = delete;
  void update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate);
  void update_partial_variables(const std::complex<FloatType> * derivativeLoss,
    const FloatType learningRate, const std::vector<int> & hiddenNodes);
  void initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates = NULL);
  void forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi);
  void forward(const std::vector<int> & spinFlipIndex, std::complex<FloatType> * lnpsi);
  void forward(const std::vector<std::vector<int> > & spinFlipIndexes, std::complex<FloatType> * lnpsi);
  void backward(std::complex<FloatType> * lnpsiGradients);
  void partial_backward(std::complex<FloatType> * lnpsiGradients, const int & nChains);
  void partial_backward(std::complex<FloatType> * lnpsiGradients, const std::vector<int> & hiddenNodes);
  void load(const FFNNDataType typeInfo, const std::string filePath);
  void save(const FFNNDataType typeInfo, const std::string filePath,
    const int precision = FloatTypeTrait_<FloatType>::precision) const;
  void spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex = -1);
  void spin_flip(const std::vector<bool> & doSpinFlip, const std::vector<std::vector<int> > & spinFlipIndexes);
  void swap_states(const int & k1, const int & k2);
  const std::complex<FloatType> * get_spinStates() const { return &spinStates_[0]; };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nHiddens() const { return knHiddens; }
  int get_nVariables() const { return variables_.size(); }
private:
  const int knInputs, knHiddens, knChains;
  const std::complex<FloatType> kzero, kone;
  const FloatType ktwo;
  const std::vector<std::complex<FloatType> > koneChains;
  const std::array<FloatType, 2> ktwoTrueFalse;
  std::vector<std::complex<FloatType> > variables_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > spinStates_;
  std::vector<std::complex<FloatType> > y_, acty_;
  std::complex<FloatType> * wi1_, * w1o_, * b1_, * d_dwi1_, * d_dw1o_, * d_db1_;
  int index_;
};

template <typename FloatType>
class FFNNTrSymm
{
public:
  FFNNTrSymm(const int nInputs, const int alpha, const int nChains);
  FFNNTrSymm(const FFNNTrSymm & rhs) = delete;
  FFNNTrSymm & operator=(const FFNNTrSymm & rhs) = delete;
  void update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate);
  void initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates = NULL);
  void forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi);
  void backward(std::complex<FloatType> * lnpsiGradients);
  void load(const std::string filePath);
  void save(const std::string filePath, const int precision = FloatTypeTrait_<FloatType>::precision) const;
  void spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex = -1);
  const std::complex<FloatType> * get_spinStates() const { return &spinStates_[0]; };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nVariables() const { return variables_.size(); }
private:
  void construct_weight_and_bias_();

  const int knInputs, kAlpha, knChains;
  const std::complex<FloatType> kzero, kone;
  const FloatType ktwo;
  const std::vector<std::complex<FloatType> > koneChains;
  const std::array<FloatType, 2> ktwoTrueFalse;
  std::vector<std::complex<FloatType> > variables_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > spinStates_;
  std::vector<std::complex<FloatType> > y_, acty_;
  std::vector<std::complex<FloatType> > wf1_, wf2_, bf_;
  std::complex<FloatType> * wi1_, * w1o_, * b1_, * d_dwi1_, * d_dw1o_, * d_db1_;
  int index_;
};


template <typename FloatType>
class FFNNSfSymm
{
public:
  FFNNSfSymm(const int nInputs, const int alpha, const int nChains);
  FFNNSfSymm(const FFNNSfSymm & rhs) = delete;
  FFNNSfSymm & operator=(const FFNNSfSymm & rhs) = delete;
  void update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate);
  void initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates = NULL);
  void forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi);
  void backward(std::complex<FloatType> * lnpsiGradients);
  void load(const std::string filePath);
  void save(const std::string filePath, const int precision = FloatTypeTrait_<FloatType>::precision) const;
  void spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex = -1);
  const std::complex<FloatType> * get_spinStates() const { return &spinStates_[0]; };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nVariables() const { return variables_.size(); }
private:
  const int knInputs, kAlpha, knChains;
  const std::complex<FloatType> kzero, kone;
  const FloatType ktwo;
  const std::vector<std::complex<FloatType> > koneChains;
  const std::array<FloatType, 2> ktwoTrueFalse;
  std::vector<std::complex<FloatType> > variables_, lnpsiGradients_;
  std::vector<std::complex<FloatType> > spinStates_;
  std::vector<std::complex<FloatType> > y_, acty_;
  std::complex<FloatType> * wi1_, * w1o_, * d_dwi1_, * d_dw1o_;
  int index_;
};
} // namespace spinhalf
#include "impl_neural_quantum_state.hpp"
