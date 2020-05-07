// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <fstream>
#include "common.cuh"
#include "cublas_template.cuh"

enum class FNNDataType { W1, W2, B1 };

template <typename FloatType>
class ComplexFNN
{
public:
  ComplexFNN(const uint32_t nInputs, const uint32_t nHiddens, const uint32_t nChains);
  ComplexFNN(const ComplexFNN & rhs) = delete;
  ComplexFNN & operator=(const ComplexFNN & rhs) = delete;
  ~ComplexFNN();
  void initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev = nullptr);
  void forward(const uint32_t spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev);
  void backward(thrust::complex<FloatType> * lnpsiGradients_dev, const thrust::host_vector<uint32_t> & hiddenNodesIdx_host);
  void update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate,
    const thrust::host_vector<uint32_t> & hiddenNodes_host);
  void spin_flip(const bool * isSpinFlipped_dev, const int32_t spinFlipIndex = -1);
  void save(const FNNDataType typeInfo, const std::string filePath, const uint32_t precision = 10u);
  void load(const FNNDataType typeInfo, const std::string filePath);
  void look_inside() const;
  thrust::complex<FloatType> * get_spinStates() { return PTR_FROM_THRUST(spinStates_dev_.data()); };
  uint32_t get_nChains() const { return knChains; }
  uint32_t get_nInputs() const { return knInputs; }
  uint32_t get_nHiddens() const { return knHiddens; }
  uint32_t get_nVariables() const { return variables_host_.size(); }
private:
  const uint32_t knInputs, knHiddens; // hyperparameters for network size
  const uint32_t knChains; // # of parallel states
  const uint32_t kgpuBlockSize1, kgpuBlockSize2, kgpuBlockSize3;
  thrust::host_vector<thrust::complex<FloatType>> variables_host_;
  thrust::device_vector<thrust::complex<FloatType>> variables_dev_;
  thrust::device_vector<thrust::complex<FloatType>> lnpsiGradients_dev_; // derivative of lnpsi with variables
  thrust::device_vector<thrust::complex<FloatType>> spinStates_dev_; // spin states (1 or -1)
  thrust::device_vector<thrust::complex<FloatType>> y_dev_, acty_dev_;
  thrust::complex<FloatType> * wi1_host_, * b1_host_, * w1o_host_; // pointer alias for weight matrix and bias vector
  thrust::complex<FloatType> * wi1_dev_, * b1_dev_, * w1o_dev_;
  thrust::complex<FloatType> * d_dwi1_dev_, * d_dw1o_dev_, * d_db1_dev_; // pointer alias for gradients
  uint32_t index_; // index of spin sites to flip
  const thrust::complex<FloatType> kzero, kone;
  const thrust::device_vector<thrust::complex<FloatType>> koneChains_dev; // [1, 1, 1,...,1]
  cublasHandle_t theCublasHandle_;
};

namespace gpu_kernel
{
template <typename FloatType>
__global__ void FNN__ActivateNeurons__(
  const uint32_t size,
  const thrust::complex<FloatType> * y,
  thrust::complex<FloatType> * acty
);

template <typename FloatType>
__global__ void FNN__ActivateNeurons__(
  const uint32_t nInputs,
  const uint32_t nHiddens,
  const uint32_t nChains,
  const uint32_t spinFlipIndex,
  const thrust::complex<FloatType> * wi1,
  const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * y,
  thrust::complex<FloatType> * acty
);

template <typename FloatType>
__global__ void FNN__ConditionalUpdateY__(
  const uint32_t nInputs,
  const uint32_t nHiddens,
  const uint32_t nChains,
  const uint32_t spinFlipIndex,
  const bool * isSpinFlipped,
  const thrust::complex<FloatType> * wi1,
  thrust::complex<FloatType> * spinStates,
  thrust::complex<FloatType> * y
);

template <typename FloatType>
__global__ void FNN__ConditionalUpdateSpin__(
  const uint32_t nInputs,
  const uint32_t nChains,
  const uint32_t spinFlipIndex,
  const bool * isSpinFlipped,
  thrust::complex<FloatType> * spinStates
);

template <typename FloatType>
__global__ void FNN__GetGradientsOfParameters__(
  const uint32_t nInputs,
  const uint32_t nHiddens,
  const uint32_t nChains,
  const uint32_t * hiddenNodesIdx,
  const uint32_t nNodes,
  const thrust::complex<FloatType> * y,
  const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * w1o,
  thrust::complex<FloatType> * d_dwi1,
  thrust::complex<FloatType> * d_db1,
  thrust::complex<FloatType> * d_dw1o
);

template <typename FloatType>
__global__ void FNN__GetlnpsiGradients__(
  const uint32_t nInputs,
  const uint32_t nHiddens,
  const uint32_t nChains,
  const uint32_t * hiddenNodesIdx,
  const uint32_t nNodes,
  const thrust::complex<FloatType> * d_dwi1,
  const thrust::complex<FloatType> * d_db1,
  const thrust::complex<FloatType> * d_dw1o,
  thrust::complex<FloatType> * lnpsiGradients
);

template <typename FloatType>
__global__ void FNN__UpdateParameters__(
  const uint32_t nInputs,
  const uint32_t nHiddens,
  const uint32_t nChains,
  const uint32_t * hiddenNodesIdx,
  const uint32_t nNodes,
  const thrust::complex<FloatType> * derivativeLoss,
  const FloatType learningRate,
  thrust::complex<FloatType> * wi1,
  thrust::complex<FloatType> * b1,
  thrust::complex<FloatType> * w1o
);
} // namespace gpu_kernel
#include "impl_neural_quantum_state.cuh"
