#pragma once

#include <omp.h>
#include <vector>
#include <complex>
#include <chrono>
#include <cstring>
#include <iomanip>
#include "blas_lapack.hpp"

typedef std::complex<double> dcomplex;
typedef std::complex<float>  scomplex;

template <typename float_t>
struct property_ {};
template <>
struct property_<float> { static constexpr int precision = 8; };
template <>
struct property_<double> { static constexpr int precision = 15; };

/* 
 * W : weight matrix
 * V : vector for visible units 
 * H : vector for hidden units
 */
enum class RBMData_t { W, V, H };

template <typename float_t>
class ComplexRBM
{
public:
  ComplexRBM(const int nInputs, const int nHiddens, const int nChains);
  ComplexRBM(const ComplexRBM & rhs) = delete;
  ComplexRBM & operator=(const ComplexRBM & rhs) = delete;
  void update_variables(const std::complex<float_t> * derivativeLoss, const float_t learningRate);
  void initialize(std::complex<float_t> * lnpsi, const std::complex<float_t> * spinStates = NULL);
  void forward(const int spinFlipIndex, std::complex<float_t> * lnpsi);
  void backward(std::complex<float_t> * lnpsiGradients);
  void load(const RBMData_t typeInfo, const std::string filePath);
  void save(const RBMData_t typeInfo, const std::string filePath, const int precision = property_<float_t>::precision) const;
  void spin_flip(const std::vector<bool> & doSpinFlip);
  const std::complex<float_t> * get_spinStates() const { return &spinStates_[0]; };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nVariables() const { return variables_.size(); }
private:
  const int knInputs, knHiddens, knChains;
  const std::vector<std::complex<float_t> > koneChains, koneHiddens;
  const std::complex<float_t> kzero, kone, ktwo;
  std::vector<std::complex<float_t> > variables_, lnpsiGradients_;
  std::vector<std::complex<float_t> > spinStates_;
  std::vector<std::complex<float_t> > y_, ly_, sa_;
  std::complex<float_t> * w_, * a_, * b_, * d_dw_, * d_da_, * d_db_;
  int index_;
};

#include "impl_ComplexRBM.hpp"
