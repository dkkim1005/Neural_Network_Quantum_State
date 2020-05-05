// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

template <typename FloatType>
ComplexFNN<FloatType>::ComplexFNN(const uint32_t nInputs, const uint32_t nHiddens, const uint32_t nChains):
  knInputs(nInputs),
  knHiddens(nHiddens),
  knChains(nChains),
  variables_host_(nInputs*nHiddens + 2u*nHiddens),
  variables_dev_(nInputs*nHiddens + 2u*nHiddens),
  lnpsiGradients_dev_(nChains*(nInputs*nHiddens + 2u*nHiddens)),
  spinStates_dev_(nInputs*nChains),
  y_dev_(nHiddens*nChains),
  acty_dev_(nHiddens*nChains),
  index_(0u),
  kzero(0.0, 0.0),
  kone(1.0, 0.0),
  koneChains_dev(nChains, thrust::complex<FloatType>(1.0, 0.0)),
  kgpuBlockSize1(CHECK_BLOCK_SIZE(1u+(nChains*nHiddens-1u)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize2(CHECK_BLOCK_SIZE(1u+(nInputs*nHiddens-1u)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize3(CHECK_BLOCK_SIZE(1u+(nChains-1u)/NUM_THREADS_PER_BLOCK))
{
  // parameter initialization: starting from the Gaussian random distribution
  uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double> randwi1(0, std::sqrt(1.0/(nInputs+nHiddens))),
    randb1(0, std::sqrt(1.0/nHiddens)), randw1o(0, std::sqrt(1.0/nHiddens));
  // host
  wi1_host_ = &variables_host_[0];
  b1_host_ = &variables_host_[nInputs*nHiddens];
  w1o_host_ = &variables_host_[nInputs*nHiddens + nHiddens];
  // device
  wi1_dev_ = PTR_FROM_THRUST(&variables_dev_[0]);
  b1_dev_ = PTR_FROM_THRUST(&variables_dev_[nInputs*nHiddens]);
  w1o_dev_ = PTR_FROM_THRUST(&variables_dev_[nInputs*nHiddens + nHiddens]);
  for (uint32_t i=0u; i<nInputs*nHiddens; ++i)
    wi1_host_[i] = thrust::complex<FloatType>(randwi1(ran), randwi1(ran));
  for (uint32_t j=0u; j<nHiddens; ++j)
    b1_host_[j] = thrust::complex<FloatType>(randb1(ran), randb1(ran));
  for (uint32_t j=0u; j<nHiddens; ++j)
    w1o_host_[j] = thrust::complex<FloatType>(randw1o(ran), randw1o(ran));
  variables_dev_ = variables_host_; // copy memory from host to device
  d_dwi1_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[0]);
  d_db1_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[nInputs*nHiddens]);
  d_dw1o_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[nInputs*nHiddens + nHiddens]);
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
}

template <typename FloatType>
ComplexFNN<FloatType>::~ComplexFNN()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename FloatType>
void ComplexFNN<FloatType>::initialize(thrust::complex<FloatType> * lnpsi_dev)
{
  thrust::fill(spinStates_dev_.begin(), spinStates_dev_.end(), kone); // spin states are initialized with 1
  // y_kj = \sum_i spinStates_ki wi1_ij + koneChains_k (x) b1_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, knHiddens, knChains, kone, b1_dev_,
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, knHiddens, knChains, knInputs, kone, kone,
    wi1_dev_, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::FNN__ActivateNeurons__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(y_dev_.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(acty_dev_.data()));  // acty_kj = ln(cosh(y_kj))
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kzero, w1o_dev_,
    PTR_FROM_THRUST(acty_dev_.data()), lnpsi_dev); // lnpsi_k = \sum_j acty_kj w1o_j
}

template <typename FloatType>
void ComplexFNN<FloatType>::forward(const uint32_t spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev)
{
  index_ = spinFlipIndex;
  gpu_kernel::FNN__ActivateNeurons__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains, index_, wi1_dev_,
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(acty_dev_.data()));
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kzero, w1o_dev_, PTR_FROM_THRUST(acty_dev_.data()), lnpsi_dev);
}

template <typename FloatType>
void ComplexFNN<FloatType>::backward(thrust::complex<FloatType> * lnpsiGradients_dev, const thrust::host_vector<uint32_t> & hiddenNodesIdx_host)
{
  /* index notation
     hiddenNodes = [j_0, j_1, j_2, ...]
     lnpsiGradients_k = [d_dwi1_k0j_0, d_dwi1_k1j_0, d_dwi1_k2j_0,... d_dwi1_k0j_1, d_dwi1_k1j_1, d_dwi1_k2j_1,...,
                         d_db1_kj_0, d_db1_kj_1, d_db1_kj_2,..., d_dw1o_kj_0, d_dw1o_kj_1, d_dw1o_kj_2,...] */
  const thrust::device_vector<uint32_t> hiddenNodesIdx_dev(hiddenNodesIdx_host);
  gpu_kernel::FNN__GetGradientsOfParameters__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains,
    PTR_FROM_THRUST(hiddenNodesIdx_dev.data()), hiddenNodesIdx_host.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(spinStates_dev_.data()), w1o_dev_, d_dwi1_dev_, d_db1_dev_, d_dw1o_dev_);
  gpu_kernel::FNN__GetlnpsiGradients__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains,
    PTR_FROM_THRUST(hiddenNodesIdx_dev.data()), hiddenNodesIdx_host.size(), d_dwi1_dev_, d_db1_dev_, d_dw1o_dev_, lnpsiGradients_dev);
}

template <typename FloatType>
void ComplexFNN<FloatType>::update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate, const thrust::host_vector<uint32_t> & hiddenNodesIdx_host)
{
  // * index notation
  // hiddenNodes = [j_0, j_1, j_2, ...]
  // derivativeLoss = [wi1_0j_0, wi1_1j_0, wi1_2j_0,... wi1_0j_1, wi1_1j_1, wi1_2j_1,..., b1_j_0, b1_j_1, b1_j_2,..., w1o_j_0, w1o_j_1, w1o_j_2]
  const thrust::device_vector<uint32_t> hiddenNodesIdx_dev(hiddenNodesIdx_host);
  gpu_kernel::FNN__UpdateParameters__<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains,
    PTR_FROM_THRUST(hiddenNodesIdx_dev.data()), hiddenNodesIdx_host.size(),
    derivativeLoss_dev, learningRate, wi1_dev_, b1_dev_, w1o_dev_);
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, knHiddens, knChains, kone, b1_dev_,
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, knHiddens, knChains, knInputs, kone, kone,
    wi1_dev_, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
}

template <typename FloatType>
void ComplexFNN<FloatType>::spin_flip(const bool * isSpinFlipped_dev, const int32_t spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  gpu_kernel::FNN__ConditionalUpdateY__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains, index_,
    isSpinFlipped_dev, wi1_dev_, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::FNN__ConditionalUpdateSpin__<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, index_,
    isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()));
}

template <typename FloatType>
void ComplexFNN<FloatType>::save(const FNNDataType typeInfo, const std::string filePath, const uint32_t precision)
{
  variables_host_ = variables_dev_; // copy memory from device to host
  // save data according to the FNNDataType
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  if (typeInfo == FNNDataType::W1)
  {
    for (uint32_t i=0u; i<knInputs; ++i)
    {
      for (uint32_t j=0u; j<knHiddens; ++j)
        writer << wi1_host_[i*knHiddens+j] << " ";
      writer << std::endl;
    }
  }
  else if (typeInfo == FNNDataType::W2)
  {
    for (uint32_t j=0u; j<knHiddens; ++j)
      writer << w1o_host_[j] << " ";
    writer << std::endl;
  }
  else if (typeInfo == FNNDataType::B1)
    for (uint32_t j=0u; j<knHiddens; ++j)
      writer << b1_host_[j] << " ";
  writer.close();
}

template <typename FloatType>
void ComplexFNN<FloatType>::load(const FNNDataType typeInfo, const std::string filePath)
{
  // read rawdata from the text file located at 'filePath'
  thrust::host_vector<thrust::complex<FloatType>> rawdata;
  std::ifstream reader(filePath);
  if (reader.is_open())
  {
    thrust::complex<FloatType> temp;
    while (reader >> temp)
      rawdata.push_back(temp);
    reader.close();
  }
  else
  {
    std::cout << "# --- file-path: " << filePath << " is not exist..." << std::endl;
    return;
  }
  // insert rawdata into 'variables_'
  if (typeInfo == FNNDataType::W1)
  {
    if (rawdata.size() == knInputs*knHiddens)
      for (uint32_t i=0u; i<rawdata.size(); ++i)
        wi1_host_[i] = rawdata[i];
    else
      std::cout << "# check 'w1' size... " << std::endl;
  }
  else if (typeInfo == FNNDataType::W2)
  {
    if (rawdata.size() == knHiddens)
      for (uint32_t i=0u; i<rawdata.size(); ++i)
        w1o_host_[i] = rawdata[i];
    else
      std::cout << "# check 'w2' size... " << std::endl;
  }
  else if (typeInfo == FNNDataType::B1)
  {
    if (rawdata.size() == knHiddens)
      for (uint32_t i=0u; i<rawdata.size(); ++i)
        b1_host_[i] = rawdata[i];
    else
      std::cout << "# check 'b1' size... " << std::endl;
  }
  variables_dev_ = variables_host_; // copy memory from host to device
}

template <typename FloatType>
void ComplexFNN<FloatType>::look_inside() const
{
  std::cout << "---- wi1" << std::flush << std::endl;
  gpu_kernel::common__Print__<<<1,1>>>(wi1_dev_, knInputs, knHiddens);
  cudaDeviceSynchronize();
  std::cout << "---- b1" << std::flush << std::endl;
  gpu_kernel::common__Print__<<<1,1>>>(b1_dev_, 1u, knHiddens);
  cudaDeviceSynchronize();
  std::cout << "---- w1o" << std::flush << std::endl;
  gpu_kernel::common__Print__<<<1,1>>>(w1o_dev_, 1u, knHiddens);
  cudaDeviceSynchronize();
}

namespace gpu_kernel
{
template <typename FloatType>
__global__ void FNN__ActivateNeurons__(
  const uint32_t size,
  const thrust::complex<FloatType> * y,
  thrust::complex<FloatType> * acty)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < size)
  {
    acty[idx] = thrust::log(thrust::cosh(y[idx]));
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void FNN__ActivateNeurons__(
  const uint32_t nInputs,
  const uint32_t nHiddens,
  const uint32_t nChains,
  const uint32_t spinFlipIndex,
  const thrust::complex<FloatType> * wi1,
  const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * y,
  thrust::complex<FloatType> * acty)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType two = 2;
  while (idx < nChains*nHiddens)
  {
    const uint32_t k = idx/nHiddens, j = idx-nHiddens*k;
    acty[idx] = thrust::log(thrust::cosh(y[idx]-wi1[spinFlipIndex*nHiddens+j]*(two*spinStates[k*nInputs+spinFlipIndex].real())));
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void FNN__ConditionalUpdateY__(
  const uint32_t nInputs,
  const uint32_t nHiddens,
  const uint32_t nChains,
  const uint32_t spinFlipIndex,
  const bool * isSpinFlipped,
  const thrust::complex<FloatType> * wi1,
  thrust::complex<FloatType> * spinStates,
  thrust::complex<FloatType> * y)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType twoDelta[2] = {0.0, 2.0};
  // update y: y_{k,j} = y_{k,j} - 2*delta(k)*wi1_{spinFlipIndex,j}*spinStates_{k,spinFlipIndex}
  while (idx < nChains*nHiddens)
  {
    const uint32_t k = idx/nHiddens, j = idx-nHiddens*k;
    y[idx] = y[idx]-wi1[spinFlipIndex*nHiddens+j]*(twoDelta[isSpinFlipped[k]]*spinStates[k*nInputs+spinFlipIndex].real());
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void FNN__ConditionalUpdateSpin__(
  const uint32_t nInputs,
  const uint32_t nChains,
  const uint32_t spinFlipIndex,
  const bool * isSpinFlipped,
  thrust::complex<FloatType> * spinStates)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType one = 1.0, twoDelta[2] = {0.0, 2.0};
  // update spin: spinState_{k,spinFlipIndex} = (1-2*delta(k))*spinStates_{k,spinFlipIndex}
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    const uint32_t k = idx;
    spinStates[k*nInputs+spinFlipIndex] = (one-twoDelta[isSpinFlipped[k]])*spinStates[k*nInputs+spinFlipIndex].real();
    idx += nstep;
  }
}

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
  thrust::complex<FloatType> * d_dw1o)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  const uint32_t varSize = nInputs*nHiddens+2u*nHiddens; // # of total variables
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x; // [k,i0] : k is the parallel chain number and i0 is the order of hiddenNodesIdx.
  // calculate gradients
  while (idx < nChains*nNodes)
  {
    const uint32_t k = idx/nNodes, i0 = idx-k*nNodes, j = hiddenNodesIdx[i0], kv = k*varSize, ki = k*nInputs, kh = k*nHiddens;
    const thrust::complex<FloatType> tany_j = thrust::tanh(y[kh+j]);
    for (uint32_t i=0u; i<nInputs; ++i)
      d_dwi1[kv+i*nHiddens+j] = tany_j*spinStates[ki+i].real()*w1o[j];
    d_db1[kv+j] = tany_j*w1o[j];
    d_dw1o[kv+j] = thrust::log(thrust::cosh(y[kh+j]));
    idx += nstep;
  }
}

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
  thrust::complex<FloatType> * lnpsiGradients)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  const uint32_t varSize = nInputs*nHiddens+2u*nHiddens; // # of total variables
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x; // [k,i0] : k is the parallel chain number and i0 is the order of hiddenNodesIdx.
  // save the results into lnpsiGradients
  const uint32_t pvarSize = nInputs*nNodes+2u*nNodes;
  while (idx < nChains*nNodes)
  {
    const uint32_t k = idx/nNodes, i0 = idx-k*nNodes, j = hiddenNodesIdx[i0], kv = k*varSize, kpv = k*pvarSize;
    for (uint32_t i=0u; i<nInputs; ++i)
      lnpsiGradients[kpv+i0*nInputs+i] = d_dwi1[kv+i*nHiddens+j];
    lnpsiGradients[kpv+nNodes*nInputs+i0] = d_db1[kv+j];
    lnpsiGradients[kpv+nNodes*(nInputs+1u)+i0] = d_dw1o[kv+j];
    idx += nstep;
  }
}

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
  thrust::complex<FloatType> * w1o)
{
  const uint32_t nstep = gridDim.x*blockDim.x;
  uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x; 
  // update wi1
  while (idx < nNodes*nInputs)
  {
    // idx : [i0,i] (i0 is the order of hiddenNodesIdx and i represents the spin site.
    const uint32_t i0 = idx/nInputs, i = idx-i0*nInputs, j = hiddenNodesIdx[i0];
    wi1[i*nHiddens+j] = wi1[i*nHiddens+j]-learningRate*derivativeLoss[idx];
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  // update b1 and w1o
  while (idx < nNodes)
  {
    // idx : [i0]
    const uint32_t j = hiddenNodesIdx[idx];
    b1[j] = b1[j]-learningRate*derivativeLoss[nNodes*nInputs+idx];
    w1o[j] = w1o[j]-learningRate*derivativeLoss[nNodes*(nInputs+1)+idx];
    idx += nstep;
  }
}
} // namespace gpu_kernel
