// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <fstream>
#include "common.cuh"
#include "cublas_template.cuh"

namespace spinhalf
{
enum class RBMDataType { W, V, H };

template <typename FloatType>
class RBM
{
public:
  RBM(const int nInputs, const int nHiddens, const int nChains);
  RBM(const RBM & rhs) = delete;
  RBM & operator=(const RBM & rhs) = delete;
  ~RBM();
  void initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev = nullptr);
  void forward(const int spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev);
  void forward(const thrust::complex<FloatType> * spinStates_dev, thrust::complex<FloatType> * lnpsi_dev, const bool saveSpinStates = true);
  void forward(const thrust::device_vector<thrust::pair<int, int> > & spinPairFlipIdx_dev, thrust::complex<FloatType> * lnpsi_dev);
  void backward(thrust::complex<FloatType> * lnpsiGradients_dev);
  void update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate);
  void spin_flip(const bool * isSpinFlipped_dev, const int spinFlipIndex = -1);
  void spin_flip(const bool * isSpinPairFlipped_dev, const thrust::device_vector<thrust::pair<int, int> > & spinPairFlipIdx_dev);
  void save(const RBMDataType typeInfo, const std::string filePath, const int precision = 10, const bool useCopyFromDeviceToHost = true);
  void save(const std::string prefix, const int precision = 10);
  void load(const RBMDataType typeInfo, const std::string filePath);
  void load(const std::string prefix);
  void copy_to(RBM<FloatType> & rhs) const;
  thrust::complex<FloatType> * get_spinStates() { return PTR_FROM_THRUST(spinStates_dev_.data()); };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nHiddens() const { return knHiddens; }
  int get_nVariables() const { return variables_host_.size(); }
private:
  const int knInputs, knHiddens; // hyperparameters of network size
  const int knChains; // # of parallel states
  const int kgpuBlockSize1, kgpuBlockSize2, kgpuBlockSize3, kgpuBlockSize4;
  thrust::host_vector<thrust::complex<FloatType>> variables_host_;
  thrust::device_vector<thrust::complex<FloatType>> variables_dev_;
  thrust::device_vector<thrust::complex<FloatType>> lnpsiGradients_dev_; // derivative of lnpsi with variables
  thrust::device_vector<thrust::complex<FloatType>> spinStates_dev_; // spin states (1 or -1)
  thrust::device_vector<thrust::complex<FloatType>> y_dev_, ly_dev_, sa_dev_;
  thrust::complex<FloatType> * w_host_, * a_host_, * b_host_; // pointer alias for weight matrix and bias vector
  thrust::complex<FloatType> * w_dev_, * a_dev_, * b_dev_;
  thrust::complex<FloatType> * d_dw_dev_, * d_da_dev_, * d_db_dev_; // pointer alias for gradients
  int index_; // index of spin sites to flip
  const thrust::complex<FloatType> kzero, kone;
  const thrust::device_vector<thrust::complex<FloatType>> koneChains_dev, koneHiddens_dev; // [1, 1, 1,...,1]
  cublasHandle_t theCublasHandle_;
};


template <typename FloatType>
class RBMTrSymm
{
public:
  RBMTrSymm(const int nInputs, const int alpha, const int nChains);
  RBMTrSymm(const RBMTrSymm & rhs) = delete;
  RBMTrSymm & operator=(const RBMTrSymm & rhs) = delete;
  ~RBMTrSymm();
  void initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev = nullptr);
  void forward(const int spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev);
  void forward(const thrust::complex<FloatType> * spinStates_dev, thrust::complex<FloatType> * lnpsi_dev, const bool saveSpinStates = true);
  void forward(const thrust::device_vector<thrust::pair<int, int> > & spinPairFlipIdx_dev, thrust::complex<FloatType> * lnpsi_dev);
  void backward(thrust::complex<FloatType> * lnpsiGradients_dev);
  void update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate);
  void spin_flip(const bool * isSpinFlipped_dev, const int spinFlipIndex = -1);
  void save(const std::string prefix, const int precision = 10);
  void load(const std::string prefix);
  void copy_to(RBMTrSymm<FloatType> & rhs) const;
  thrust::complex<FloatType> * get_spinStates() { return PTR_FROM_THRUST(spinStates_dev_.data()); };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_alpha() const { return kAlpha; }
  int get_nVariables() const { return variables_host_.size(); }
private:
  void symmetrize_variables_();
  const int knInputs, kAlpha; // hyperparameters of network size
  const int knChains; // # of parallel states
  const int kgpuBlockSize1, kgpuBlockSize2, kgpuBlockSize3, kgpuBlockSize4;
  thrust::host_vector<thrust::complex<FloatType>> variables_host_;
  thrust::device_vector<thrust::complex<FloatType>> variables_dev_;
  thrust::device_vector<thrust::complex<FloatType>> lnpsiGradients_dev_; // derivative of lnpsi with variables
  thrust::device_vector<thrust::complex<FloatType>> spinStates_dev_; // spin states (1 or -1)
  thrust::device_vector<thrust::complex<FloatType>> y_dev_, ly_dev_, sa_dev_;
  thrust::complex<FloatType> * w_host_, * a_host_, * b_host_; // pointer alias for weight matrix and bias vector
  thrust::complex<FloatType> * w_dev_, * a_dev_, * b_dev_;
  thrust::device_vector<thrust::complex<FloatType>> wf_dev_, af_dev_, bf_dev_;
  thrust::complex<FloatType> * d_dw_dev_, * d_da_dev_, * d_db_dev_; // pointer alias for gradients
  int index_; // index of spin sites to flip
  const thrust::complex<FloatType> kzero, kone;
  const thrust::device_vector<thrust::complex<FloatType>> koneChains_dev, koneHiddens_dev; // [1, 1, 1,...,1]
  cublasHandle_t theCublasHandle_;
};


template <typename FloatType>
class RBMZ2PrSymm
{
public:
  RBMZ2PrSymm(const int nInputs, const int alpha, const int nChains);
  RBMZ2PrSymm(const RBMZ2PrSymm & rhs) = delete;
  RBMZ2PrSymm & operator=(const RBMZ2PrSymm & rhs) = delete;
  ~RBMZ2PrSymm();
  void initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev = nullptr);
  void forward(const int spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev);
  void forward(const thrust::complex<FloatType> * spinStates_dev, thrust::complex<FloatType> * lnpsi_dev, const bool saveSpinStates = true);
  void backward(thrust::complex<FloatType> * lnpsiGradients_dev);
  void update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate);
  void spin_flip(const bool * isSpinFlipped_dev, const int spinFlipIndex = -1);
  void save(const std::string prefix, const int precision = 10);
  void load(const std::string prefix);
  void copy_to(RBMZ2PrSymm<FloatType> & rhs) const;
  thrust::complex<FloatType> * get_spinStates() { return PTR_FROM_THRUST(spinStates_dev_.data()); };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_alpha() const { return kAlpha; }
  int get_nVariables() const { return variables_host_.size(); }
private:
  void symmetrize_variables_();
  const int knInputs, kAlpha; // hyperparameters of network size
  const int knChains; // # of parallel states
  const int kgpuBlockSize1, kgpuBlockSize2, kgpuBlockSize3, kgpuBlockSize4;
  thrust::host_vector<thrust::complex<FloatType>> variables_host_;
  thrust::device_vector<thrust::complex<FloatType>> variables_dev_;
  thrust::device_vector<thrust::complex<FloatType>> lnpsiGradients_dev_; // derivative of lnpsi with variables
  thrust::device_vector<thrust::complex<FloatType>> spinStates_dev_; // spin states (1 or -1)
  thrust::device_vector<thrust::complex<FloatType>> y_dev_, ly_dev_;
  thrust::complex<FloatType> * w_host_, * b_host_; // pointer alias for weight matrix and bias vector
  thrust::complex<FloatType> * w_dev_, * b_dev_;
  thrust::device_vector<thrust::complex<FloatType>> wf_dev_, bf_dev_;
  thrust::complex<FloatType> * d_dw_dev_, * d_db_dev_; // pointer alias for gradients
  int index_; // index of spin sites to flip
  const thrust::complex<FloatType> kzero, kone;
  const thrust::device_vector<thrust::complex<FloatType>> koneChains_dev, koneHiddens_dev; // [1, 1, 1,...,1]
  cublasHandle_t theCublasHandle_;
};


enum class FFNNDataType { W1, W2, B1 };

template <typename FloatType>
class FFNN
{
public:
  FFNN(const int nInputs, const int nHiddens, const int nChains);
  FFNN(const FFNN & rhs) = delete;
  FFNN & operator=(const FFNN & rhs) = delete;
  ~FFNN();
  void initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev = nullptr);
  void forward(const int spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev);
  void forward(const thrust::complex<FloatType> * spinStates_dev, thrust::complex<FloatType> * lnpsi_dev, const bool saveSpinStates = true);
  void forward(const thrust::device_vector<thrust::pair<int, int> > & spinPairFlipIdx_dev, thrust::complex<FloatType> * lnpsi_dev);
  void backward(thrust::complex<FloatType> * lnpsiGradients_dev);
  void update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate);
  void spin_flip(const bool * isSpinFlipped_dev, const int spinFlipIndex = -1);
  void spin_flip(const bool * isSpinFlipped_dev, const thrust::device_vector<thrust::pair<int, int> > & spinPairFlipIdx_dev);
  void save(const FFNNDataType typeInfo, const std::string filePath, const int precision = 10, const bool useCopyFromDeviceToHost = true);
  void save(const std::string filePath, const int precision = 10);
  void load(const FFNNDataType typeInfo, const std::string filePath);
  void load(const std::string prefix);
  void copy_to(FFNN<FloatType> & rhs) const;
  void look_inside() const;
  thrust::complex<FloatType> * get_spinStates() { return PTR_FROM_THRUST(spinStates_dev_.data()); };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_nHiddens() const { return knHiddens; }
  int get_nVariables() const { return variables_host_.size(); }
private:
  const int knInputs, knHiddens; // hyperparameters of network size
  const int knChains; // # of parallel states
  const int kgpuBlockSize1, kgpuBlockSize2, kgpuBlockSize3;
  thrust::host_vector<thrust::complex<FloatType>> variables_host_;
  thrust::device_vector<thrust::complex<FloatType>> variables_dev_;
  thrust::device_vector<thrust::complex<FloatType>> lnpsiGradients_dev_; // derivative of lnpsi with variables
  thrust::device_vector<thrust::complex<FloatType>> spinStates_dev_; // spin states (1 or -1)
  thrust::device_vector<thrust::complex<FloatType>> y_dev_, acty_dev_;
  thrust::complex<FloatType> * wi1_host_, * b1_host_, * w1o_host_; // pointer alias for weight matrix and bias vector
  thrust::complex<FloatType> * wi1_dev_, * b1_dev_, * w1o_dev_;
  thrust::complex<FloatType> * d_dwi1_dev_, * d_dw1o_dev_, * d_db1_dev_; // pointer alias for gradients
  int index_; // index of spin sites to flip
  const thrust::complex<FloatType> kzero, kone;
  const thrust::device_vector<thrust::complex<FloatType>> koneChains_dev; // [1, 1, 1,...,1]
  cublasHandle_t theCublasHandle_;
};


template <typename FloatType>
class FFNNTrSymm
{
public:
  FFNNTrSymm(const int nInputs, const int alpha, const int nChains);
  FFNNTrSymm(const FFNNTrSymm & rhs) = delete;
  FFNNTrSymm & operator=(const FFNNTrSymm & rhs) = delete;
  ~FFNNTrSymm();
  void initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev = nullptr);
  void forward(const int spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev);
  void forward(const thrust::complex<FloatType> * spinStates_dev, thrust::complex<FloatType> * lnpsi_dev, const bool saveSpinStates = true);
  void backward(thrust::complex<FloatType> * lnpsiGradients_dev);
  void update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate);
  void spin_flip(const bool * isSpinFlipped_dev, const int spinFlipIndex = -1);
  void save(const std::string filePath, const int precision = 10);
  void load(const std::string filePath);
  void copy_to(FFNNTrSymm<FloatType> & rhs) const;
  thrust::complex<FloatType> * get_spinStates() { return PTR_FROM_THRUST(spinStates_dev_.data()); };
  int get_nChains() const { return knChains; }
  int get_nInputs() const { return knInputs; }
  int get_alpha() const { return kAlpha; }
  int get_nVariables() const { return variables_host_.size(); }
private:
  void symmetrize_variables_();
  const int knInputs, kAlpha; // hyperparameters of network size
  const int knChains; // # of parallel states
  const int kgpuBlockSize1, kgpuBlockSize2, kgpuBlockSize3;
  thrust::host_vector<thrust::complex<FloatType>> variables_host_;
  thrust::device_vector<thrust::complex<FloatType>> variables_dev_;
  thrust::device_vector<thrust::complex<FloatType>> lnpsiGradients_dev_; // derivative of lnpsi with variables
  thrust::device_vector<thrust::complex<FloatType>> spinStates_dev_; // spin states (1 or -1)
  thrust::device_vector<thrust::complex<FloatType>> y_dev_, acty_dev_;
  thrust::complex<FloatType> * wi1_host_, * b1_host_, * w1o_host_; // pointer alias for weight matrix and bias vector
  thrust::complex<FloatType> * wi1_dev_, * b1_dev_, * w1o_dev_;
  thrust::device_vector<thrust::complex<FloatType>> wi1f_dev_, b1f_dev_, w1of_dev_;
  thrust::complex<FloatType> * d_dwi1_dev_, * d_dw1o_dev_, * d_db1_dev_; // pointer alias for gradients
  int index_; // index of spin sites to flip
  const thrust::complex<FloatType> kzero, kone;
  const thrust::device_vector<thrust::complex<FloatType>> koneChains_dev; // [1, 1, 1,...,1]
  cublasHandle_t theCublasHandle_;
};

template <typename FloatType>
void generate_random_binary_dist(thrust::device_vector<thrust::complex<FloatType> > & vector_dev)
{
  thrust::host_vector<thrust::complex<FloatType> > vector_host(vector_dev.size());
  std::mt19937_64 rand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<FloatType> dist;
  constexpr FloatType half = 0.5, one = 1.0;
  for (auto & item : vector_host)
    item = ((dist(rand) > half) ? one : -one);
  vector_dev = vector_host;
}
} // end namespace spinhalf


namespace gpu_device
{
__constant__ float kln2f[1];
__constant__ double kln2d[1];
__device__ thrust::complex<float> logcosh(const thrust::complex<float> z);
__device__ thrust::complex<double> logcosh(const thrust::complex<double> z);
}

namespace gpu_kernel
{
// GPU kernels for common uses
template <typename FloatType>
__global__ void logcosh(const int size, const thrust::complex<FloatType> * y, thrust::complex<FloatType> * z);

template <typename FloatType>
__global__ void logcosh(const int nInputs, const int nHiddens, const int nChains, const int spinFlipIndex,
  const thrust::complex<FloatType> * w, const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * y, thrust::complex<FloatType> * z);

template <typename FloatType>
__global__ void logcosh(const int nInputs, const int nHiddens, const int nChains,
  const thrust::pair<int, int> * spinPairFlipIdx,
  const thrust::complex<FloatType> * wi1, const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * y, thrust::complex<FloatType> * z);

template <typename FloatType>
__global__ void update_parameters(const int size, const thrust::complex<FloatType> * derivativeLoss,
  const FloatType learningRate, thrust::complex<FloatType> * variables);

template <typename FloatType>
__global__ void conditional_y_update(const int nInputs, const int nHiddens, const int nChains,
  const int spinFlipIndex, const bool * isSpinFlipped, const thrust::complex<FloatType> * wi1,
  thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * y);

template <typename FloatType>
__global__ void conditional_y_update(const int nInputs, const int nHiddens, const int nChains,
  const thrust::pair<int, int> * spinPairFlipIdx, const bool * isSpinFlipped, const thrust::complex<FloatType> * w,
  thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * y);

template <typename FloatType>
__global__ void conditional_spin_update(const int nInputs, const int nChains, const int spinFlipIndex,
  const bool * isSpinFlipped, thrust::complex<FloatType> * spinStates);

template <typename FloatType>
__global__ void conditional_spin_update(const int nInputs, const int nChains, const thrust::pair<int, int> * spinPairFlipIdx,
  const bool * isSpinFlipped, thrust::complex<FloatType> * spinStates);


// GPU kernels for RBM, RBMTrSymm, and RBMZ2PrSymm
template <typename FloatType>
__global__ void RBM__sadot__(const int nInputs, const int nChains, const int spinFlipIndex,
  const thrust::complex<FloatType> * sa, const thrust::complex<FloatType> * a,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * lnpsi);

template <typename FloatType>
__global__ void RBM__sadot__(const int nInputs, const int nChains, const thrust::pair<int, int> * spinPairFlipIdx,
  const thrust::complex<FloatType> * sa, const thrust::complex<FloatType> * a,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * lnpsi);

template <typename FloatType>
__global__ void RBM__GetGradientsOfParameters__(const int nInputs, const int nHiddens, const int nChains,
  const thrust::complex<FloatType> * y, const thrust::complex<FloatType> * spinStates,
  thrust::complex<FloatType> * d_dw, thrust::complex<FloatType> * d_da, thrust::complex<FloatType> * d_db);

template <typename FloatType>
__global__ void RBM__saUpdate__(const int nInputs, const int nChains, const int spinFlipIndex,
  const bool * isSpinFlipped, const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * a, thrust::complex<FloatType> * sa);

template <typename FloatType>
__global__ void RBM__saUpdate__(const int nInputs, const int nChains, const thrust::pair<int, int> * spinPairFlipIdx,
  const bool * isSpinFlipped, const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * a, thrust::complex<FloatType> * sa);

template <typename FloatType>
__global__ void RBMTrSymm__GetGradientsOfParameters__(const int nInputs, const int alpha, const int nChains,
  const thrust::complex<FloatType> * y, const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * d_dw,
  thrust::complex<FloatType> * d_da, thrust::complex<FloatType> * d_db);

template <typename FloatType>
__global__ void RBMTrSymm__ConstructWeightAndBias__(const int alpha, const int nInputs,
  const thrust::complex<FloatType> * w, const thrust::complex<FloatType> * a, const thrust::complex<FloatType> * b,
  thrust::complex<FloatType> * wf, thrust::complex<FloatType> * af, thrust::complex<FloatType> * bf);

template <typename FloatType>
__global__ void RBMZ2PrSymm__GetGradientsOfParameters__(const int nInputs, const int alpha, const int nChains,
  const thrust::complex<FloatType> * y, const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * d_dw,
  thrust::complex<FloatType> * d_db);

template <typename FloatType>
__global__ void RBMZ2PrSymm__ConstructWeightAndBias__(const int alpha, const int nInputs,
  const thrust::complex<FloatType> * w, const thrust::complex<FloatType> * b,
  thrust::complex<FloatType> * wf, thrust::complex<FloatType> * bf);


// GPU kernels for FFNN and FFNNTrSymm
template <typename FloatType>
__global__ void FFNN__GetGradientsOfParameters__(const int nInputs, const int nHiddens, const int nChains,
  const thrust::complex<FloatType> * y, const thrust::complex<FloatType> * spinStates, const thrust::complex<FloatType> * w1o,
  thrust::complex<FloatType> * d_dwi1, thrust::complex<FloatType> * d_db1, thrust::complex<FloatType> * d_dw1o);

template <typename FloatType>
__global__ void FFNN__GetlnpsiGradients__(const int nInputs, const int nHiddens, const int nChains,
  const thrust::complex<FloatType> * d_dwi1, const thrust::complex<FloatType> * d_db1, const thrust::complex<FloatType> * d_dw1o,
  thrust::complex<FloatType> * lnpsiGradients);

template <typename FloatType>
__global__ void FFNN__UpdateParameters__(const int nInputs, const int nHiddens, const int nChains,
  const thrust::complex<FloatType> * derivativeLoss, const FloatType learningRate, thrust::complex<FloatType> * wi1,
  thrust::complex<FloatType> * b1, thrust::complex<FloatType> * w1o);

template <typename FloatType>
__global__ void FFNNTrSymm__ConstructWeightAndBias__(const int alpha, const int nInputs,
  const thrust::complex<FloatType> * wi1, const thrust::complex<FloatType> * b1, const thrust::complex<FloatType> * w1o,
  thrust::complex<FloatType> * wi1f, thrust::complex<FloatType> * b1f, thrust::complex<FloatType> * w1of);

template <typename FloatType>
__global__ void FFNNTrSymm__GetGradientsOfParameters__(const int nInputs, const int alpha, const int nChains,
  const thrust::complex<FloatType> * y, const thrust::complex<FloatType> * spinStates, const thrust::complex<FloatType> * w1of,
  thrust::complex<FloatType> * d_dwi1, thrust::complex<FloatType> * d_db1, thrust::complex<FloatType> * d_dw1o);
} // namespace gpu_kernel
#include "impl_neural_quantum_state.cuh"
