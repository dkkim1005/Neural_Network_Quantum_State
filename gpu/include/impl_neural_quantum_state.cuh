// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

namespace spinhalf
{
template <typename FloatType>
RBM<FloatType>::RBM(const int nInputs, const int nHiddens, const int nChains):
  knInputs(nInputs),
  knHiddens(nHiddens),
  knChains(nChains),
  variables_host_(nInputs*nHiddens+nInputs+nHiddens),
  variables_dev_(nInputs*nHiddens+nInputs+nHiddens),
  lnpsiGradients_dev_(nChains*(nInputs*nHiddens+nInputs+nHiddens)),
  spinStates_dev_(nInputs*nChains),
  y_dev_(nHiddens*nChains),
  ly_dev_(nHiddens*nChains),
  sa_dev_(nInputs*nChains),
  index_(0),
  kzero(0.0, 0.0),
  kone(1.0, 0.0),
  koneChains_dev(nChains, thrust::complex<FloatType>(1.0, 0.0)),
  koneHiddens_dev(nHiddens, thrust::complex<FloatType>(1.0, 0.0)),
  kgpuBlockSize1(CHECK_BLOCK_SIZE(1+(nChains*nHiddens-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize2(CHECK_BLOCK_SIZE(1+(nInputs*nHiddens-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize3(CHECK_BLOCK_SIZE(1+(nChains-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize4(CHECK_BLOCK_SIZE(1+(nChains*((nInputs>nHiddens) ? nInputs:nHiddens)-1)/NUM_THREADS_PER_BLOCK))
{
  // parameter initialization: starting from the Gaussian random distribution
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double>
    randw(0, std::sqrt(1.0/(nInputs+nHiddens))),
    randb(0, std::sqrt(1.0/nHiddens));
  // host
  w_host_ = &variables_host_[0];
  a_host_ = &variables_host_[nInputs*nHiddens];
  b_host_ = &variables_host_[nInputs*nHiddens+nInputs];
  // device
  w_dev_ = PTR_FROM_THRUST(&variables_dev_[0]);
  a_dev_ = PTR_FROM_THRUST(&variables_dev_[nInputs*nHiddens]);
  b_dev_ = PTR_FROM_THRUST(&variables_dev_[nInputs*nHiddens+nInputs]);
  for (int i=0; i<nInputs*nHiddens; ++i)
    w_host_[i] = thrust::complex<FloatType>(1e-1*randw(ran), 1e-1*randw(ran));
  for (int i=0; i<nInputs; ++i)
    a_host_[i] = kzero;
  for (int j=0; j<nHiddens; ++j)
    b_host_[j] = thrust::complex<FloatType>(1e-1*randb(ran), 1e-1*randb(ran));
  variables_dev_ = variables_host_; // copy memory from host to device
  d_dw_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[0]);
  d_da_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[nInputs*nHiddens]);
  d_db_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[nInputs*nHiddens+nInputs]);
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
  const float ln2f = std::log(2.0f);
  const double ln2d = std::log(2.0);
  CHECK_ERROR(cudaSuccess, cudaMemcpyToSymbol(gpu_device::kln2f, &ln2f, sizeof(float)));
  CHECK_ERROR(cudaSuccess, cudaMemcpyToSymbol(gpu_device::kln2d, &ln2d, sizeof(double)));
}

template <typename FloatType>
RBM<FloatType>::~RBM()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename FloatType>
void RBM<FloatType>::initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev)
{
  if (spinStates_dev == nullptr)
    thrust::fill(spinStates_dev_.begin(), spinStates_dev_.end(), kone); // spin states are initialized with 1
  else
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(spinStates_dev_.data()), spinStates_dev,
      sizeof(thrust::complex<FloatType>)*spinStates_dev_.size(), cudaMemcpyDeviceToDevice));
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, knHiddens, knChains, kone, b_dev_,
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, knHiddens, knChains, knInputs, kone, kone,
    w_dev_, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(y_dev_.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(ly_dev_.data()));  // ly_kj = ln(cosh(y_kj))
  // sa_k = \sum_i a_i*spinStates_ki
  cublas::gemm(theCublasHandle_, 1, knChains, knInputs, kone, kzero, PTR_FROM_THRUST(a_dev_),
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(sa_dev_.data()));
  // lnpsi_k = \sum_j ly_kj + sa_k
  CHECK_ERROR(cudaSuccess, cudaMemcpy(lnpsi_dev, PTR_FROM_THRUST(sa_dev_.data()),
    sizeof(thrust::complex<FloatType>)*knChains, cudaMemcpyDeviceToDevice));
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kone, PTR_FROM_THRUST(koneHiddens_dev.data()),
    PTR_FROM_THRUST(ly_dev_.data()), lnpsi_dev);
}

template <typename FloatType>
void RBM<FloatType>::forward(const int spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev)
{
  index_ = spinFlipIndex;
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains, index_, w_dev_,
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(ly_dev_.data()));
  // lnpsi_k = \sum_j ly_kj + \sum_i a_i*spinStates_ki
  gpu_kernel::RBM__sadot__<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, spinFlipIndex, PTR_FROM_THRUST(sa_dev_.data()),
    PTR_FROM_THRUST(a_dev_), PTR_FROM_THRUST(spinStates_dev_.data()), lnpsi_dev);
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kone, PTR_FROM_THRUST(koneHiddens_dev.data()),
    PTR_FROM_THRUST(ly_dev_.data()), lnpsi_dev);
}

template <typename FloatType>
void RBM<FloatType>::forward(const thrust::complex<FloatType> * spinStates_dev,
  thrust::complex<FloatType> * lnpsi_dev, const bool saveSpinStates)
{
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, knHiddens, knChains, kone, b_dev_,
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, knHiddens, knChains, knInputs, kone, kone,
    w_dev_, spinStates_dev, PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(y_dev_.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(ly_dev_.data()));  // ly_kj = ln(cosh(y_kj))
  // sa_k = \sum_i a_i*spinStates_ki
  cublas::gemm(theCublasHandle_, 1, knChains, knInputs, kone, kzero, PTR_FROM_THRUST(a_dev_),
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(sa_dev_.data()));
  // lnpsi_k = \sum_j ly_kj + sa_k
  CHECK_ERROR(cudaSuccess, cudaMemcpy(lnpsi_dev, PTR_FROM_THRUST(sa_dev_.data()),
    sizeof(thrust::complex<FloatType>)*knChains, cudaMemcpyDeviceToDevice));
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kone, PTR_FROM_THRUST(koneHiddens_dev.data()),
    PTR_FROM_THRUST(ly_dev_.data()), lnpsi_dev);
  if (saveSpinStates)
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(spinStates_dev_.data()), spinStates_dev,
      sizeof(thrust::complex<FloatType>)*spinStates_dev_.size(), cudaMemcpyDeviceToDevice));
}

template <typename FloatType>
void RBM<FloatType>::forward(const thrust::device_vector<thrust::pair<int, int> > & spinPairFlipIdx_dev,
  thrust::complex<FloatType> * lnpsi_dev)
{
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains,
    PTR_FROM_THRUST(spinPairFlipIdx_dev.data()), w_dev_, PTR_FROM_THRUST(spinStates_dev_.data()),
    PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(ly_dev_.data()));
  // lnpsi_k = \sum_j ly_kj + \sum_i a_i*spinStates_ki
  gpu_kernel::RBM__sadot__<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains,
    PTR_FROM_THRUST(spinPairFlipIdx_dev.data()), PTR_FROM_THRUST(sa_dev_.data()),
    PTR_FROM_THRUST(a_dev_), PTR_FROM_THRUST(spinStates_dev_.data()), lnpsi_dev);
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kone, PTR_FROM_THRUST(koneHiddens_dev.data()),
    PTR_FROM_THRUST(ly_dev_.data()), lnpsi_dev);
}

template <typename FloatType>
void RBM<FloatType>::backward(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  gpu_kernel::RBM__GetGradientsOfParameters__<<<kgpuBlockSize4, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains,
    PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()),
    PTR_FROM_THRUST(d_dw_dev_), PTR_FROM_THRUST(d_da_dev_), PTR_FROM_THRUST(d_db_dev_));
  CHECK_ERROR(cudaSuccess, cudaMemcpy(lnpsiGradients_dev, PTR_FROM_THRUST(lnpsiGradients_dev_.data()),
    sizeof(std::complex<FloatType>)*variables_dev_.size()*knChains, cudaMemcpyDeviceToDevice));
}

template <typename FloatType>
void RBM<FloatType>::update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate)
{
  gpu_kernel::update_parameters<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(variables_dev_.size(),
    derivativeLoss_dev, learningRate, PTR_FROM_THRUST(variables_dev_.data()));
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, knHiddens, knChains, kone, b_dev_,
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, knHiddens, knChains, knInputs, kone, kone,
    w_dev_, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  // sa_k = \sum_i a_i*spinStates_ki
  cublas::gemm(theCublasHandle_, 1, knChains, knInputs, kone, kzero, PTR_FROM_THRUST(a_dev_),
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(sa_dev_.data()));
}

template <typename FloatType>
void RBM<FloatType>::spin_flip(const bool * isSpinFlipped_dev, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  gpu_kernel::conditional_y_update<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains, index_,
    isSpinFlipped_dev, w_dev_, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::RBM__saUpdate__<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, index_,
    isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(a_dev_), PTR_FROM_THRUST(sa_dev_.data()));
  gpu_kernel::conditional_spin_update<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, index_,
    isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()));
}

template <typename FloatType>
void RBM<FloatType>::spin_flip(const bool * isSpinFlipped_dev, const thrust::device_vector<thrust::pair<int, int> > & spinPairFlipIdx_dev)
{
  gpu_kernel::conditional_y_update<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains,
    PTR_FROM_THRUST(spinPairFlipIdx_dev.data()), isSpinFlipped_dev, w_dev_,
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::RBM__saUpdate__<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, PTR_FROM_THRUST(spinPairFlipIdx_dev.data()),
    isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(a_dev_), PTR_FROM_THRUST(sa_dev_.data()));
  gpu_kernel::conditional_spin_update<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains,
    PTR_FROM_THRUST(spinPairFlipIdx_dev.data()), isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()));
}

template <typename FloatType>
void RBM<FloatType>::save(const RBMDataType typeInfo, const std::string filePath, const int precision, const bool useCopyFromDeviceToHost)
{
  if (useCopyFromDeviceToHost)
    variables_host_ = variables_dev_; // copy memory from device to host
  // save data according to the RBMDataType
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  if (typeInfo == RBMDataType::W)
  {
    for (int i=0; i<knInputs; ++i)
    {
      for (int j=0; j<knHiddens; ++j)
        writer << w_host_[i*knHiddens+j] << " ";
      writer << std::endl;
    }
  }
  else if (typeInfo == RBMDataType::V)
  {
    for (int i=0; i<knInputs; ++i)
      writer << a_host_[i] << " ";
    writer << std::endl;
  }
  else if (typeInfo == RBMDataType::H)
    for (int j=0; j<knHiddens; ++j)
      writer << b_host_[j] << " ";
  writer.close();
}

template <typename FloatType>
void RBM<FloatType>::save(const std::string prefix, const int precision)
{
  variables_host_ = variables_dev_; // copy memory from device to host
  this->save(RBMDataType::W, prefix + "Dw.dat", precision, false);
  this->save(RBMDataType::V, prefix + "Da.dat", precision, false);
  this->save(RBMDataType::H, prefix + "Db.dat", precision, false);
}

template <typename FloatType>
void RBM<FloatType>::load(const RBMDataType typeInfo, const std::string filePath)
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
  if (typeInfo == RBMDataType::W)
  {
    if (rawdata.size() == knInputs*knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        w_host_[i] = rawdata[i];
    else
      std::cout << "# check 'w' size... " << std::endl;
  }
  else if (typeInfo == RBMDataType::V)
  {
    if (rawdata.size() == knInputs)
      for (int i=0; i<rawdata.size(); ++i)
        a_host_[i] = rawdata[i];
    else
      std::cout << "# check 'a' size... " << std::endl;
  }
  else if (typeInfo == RBMDataType::H)
  {
    if (rawdata.size() == knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        b_host_[i] = rawdata[i];
    else
      std::cout << "# check 'b' size... " << std::endl;
  }
  variables_dev_ = variables_host_; // copy memory from host to device
}

template <typename FloatType>
void RBM<FloatType>::load(const std::string prefix)
{
  this->load(RBMDataType::W, prefix + "Dw.dat");
  this->load(RBMDataType::V, prefix + "Da.dat");
  this->load(RBMDataType::H, prefix + "Db.dat");
}

template <typename FloatType>
void RBM<FloatType>::copy_to(RBM<FloatType> & rbm) const
{
  if (knChains != rbm.get_nChains())
    throw std::length_error("knChains != rbm.get_nChains()");
  if (knInputs != rbm.get_nInputs())
    throw std::length_error("knInputs != rbm.get_nInputs()");
  if (knHiddens != rbm.get_nHiddens())
    throw std::length_error("knHiddens != rbm.get_nHiddens()");
  rbm.variables_dev_ = variables_dev_;
}


template <typename FloatType>
RBMTrSymm<FloatType>::RBMTrSymm(const int nInputs, const int alpha, const int nChains):
  knInputs(nInputs),
  kAlpha(alpha),
  knChains(nChains),
  variables_host_(nInputs*alpha+1+alpha),
  variables_dev_(nInputs*alpha+1+alpha),
  lnpsiGradients_dev_(nChains*(nInputs*alpha+1+alpha)),
  spinStates_dev_(nInputs*nChains),
  y_dev_(nInputs*alpha*nChains),
  ly_dev_(nInputs*alpha*nChains),
  sa_dev_(nChains),
  wf_dev_(nInputs*nInputs*alpha),
  af_dev_(nInputs),
  bf_dev_(nInputs*alpha),
  index_(0),
  kzero(0.0, 0.0),
  kone(1.0, 0.0),
  koneChains_dev(nChains, thrust::complex<FloatType>(1.0, 0.0)),
  koneHiddens_dev(nInputs*alpha, thrust::complex<FloatType>(1.0, 0.0)),
  kgpuBlockSize1(CHECK_BLOCK_SIZE(1+(nChains*nInputs*alpha-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize2(CHECK_BLOCK_SIZE(1+(nInputs*nInputs*alpha-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize3(CHECK_BLOCK_SIZE(1+(nChains-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize4(CHECK_BLOCK_SIZE(1+(nInputs*alpha+alpha)/NUM_THREADS_PER_BLOCK))
{
  // parameter initialization: starting from the Gaussian random distribution
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double>
    randw(0, std::sqrt(1.0/((1+kAlpha)*knInputs))),
    randa(0, std::sqrt(1.0/knInputs)),
    randb(0, std::sqrt(1.0/(knInputs*kAlpha)));
  // host
  w_host_ = &variables_host_[0];
  a_host_ = &variables_host_[knInputs*kAlpha];
  b_host_ = &variables_host_[knInputs*kAlpha+1];
  // device
  w_dev_ = PTR_FROM_THRUST(&variables_dev_[0]);
  a_dev_ = PTR_FROM_THRUST(&variables_dev_[knInputs*kAlpha]);
  b_dev_ = PTR_FROM_THRUST(&variables_dev_[knInputs*kAlpha+1]);
  for (int i=0; i<knInputs*kAlpha; ++i)
    w_host_[i] = thrust::complex<FloatType>(1e-1*randw(ran), 1e-1*randw(ran));
  a_host_[0] = kzero;
  for (int j=0; j<kAlpha; ++j)
    b_host_[j] = thrust::complex<FloatType>(1e-1*randb(ran), 1e-1*randb(ran));
  variables_dev_ = variables_host_; // copy memory from host to device
  d_dw_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[0]);
  d_da_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[knInputs*kAlpha]);
  d_db_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[knInputs*kAlpha+1]);
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
  const float ln2f = std::log(2.0f);
  const double ln2d = std::log(2.0);
  CHECK_ERROR(cudaSuccess, cudaMemcpyToSymbol(gpu_device::kln2f, &ln2f, sizeof(float)));
  CHECK_ERROR(cudaSuccess, cudaMemcpyToSymbol(gpu_device::kln2d, &ln2d, sizeof(double)));
}

template <typename FloatType>
RBMTrSymm<FloatType>::~RBMTrSymm()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename FloatType>
void RBMTrSymm<FloatType>::initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev)
{
  if (spinStates_dev == nullptr)
    thrust::fill(spinStates_dev_.begin(), spinStates_dev_.end(), kone); // spin states are initialized with 1
  else
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(spinStates_dev_.data()), spinStates_dev,
      sizeof(thrust::complex<FloatType>)*spinStates_dev_.size(), cudaMemcpyDeviceToDevice));
  this->symmetrize_variables();
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, kAlpha*knInputs, knChains, kone, PTR_FROM_THRUST(bf_dev_.data()),
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, kAlpha*knInputs, knChains, knInputs, kone, kone,
    PTR_FROM_THRUST(wf_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(y_dev_.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(ly_dev_.data()));  // ly_kj = ln(cosh(y_kj))
  // sa_k = \sum_i af_i*spinStates_ki
  cublas::gemm(theCublasHandle_, 1, knChains, knInputs, kone, kzero, PTR_FROM_THRUST(af_dev_.data()),
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(sa_dev_.data()));
  // lnpsi_k = \sum_j ly_kj + sa_k
  CHECK_ERROR(cudaSuccess, cudaMemcpy(lnpsi_dev, PTR_FROM_THRUST(sa_dev_.data()),
    sizeof(thrust::complex<FloatType>)*knChains, cudaMemcpyDeviceToDevice));
  cublas::gemm(theCublasHandle_, 1, knChains, kAlpha*knInputs, kone, kone, PTR_FROM_THRUST(koneHiddens_dev.data()),
    PTR_FROM_THRUST(ly_dev_.data()), lnpsi_dev);
}

template <typename FloatType>
void RBMTrSymm<FloatType>::forward(const int spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev)
{
  index_ = spinFlipIndex;
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, kAlpha*knInputs, knChains, index_,
    PTR_FROM_THRUST(wf_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(ly_dev_.data()));
  // lnpsi_k = \sum_j ly_kj + \sum_i a_i*spinStates_ki
  gpu_kernel::RBM__sadot__<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, spinFlipIndex, PTR_FROM_THRUST(sa_dev_.data()),
    PTR_FROM_THRUST(af_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), lnpsi_dev);
  cublas::gemm(theCublasHandle_, 1, knChains, kAlpha*knInputs, kone, kone, PTR_FROM_THRUST(koneHiddens_dev.data()),
    PTR_FROM_THRUST(ly_dev_.data()), lnpsi_dev);
}

template <typename FloatType>
void RBMTrSymm<FloatType>::forward(const thrust::complex<FloatType> * spinStates_dev,
  thrust::complex<FloatType> * lnpsi_dev, const bool saveSpinStates)
{
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, kAlpha*knInputs, knChains, kone, PTR_FROM_THRUST(bf_dev_.data()),
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, kAlpha*knInputs, knChains, knInputs, kone, kone,
    PTR_FROM_THRUST(wf_dev_.data()), spinStates_dev, PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(y_dev_.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(ly_dev_.data()));  // ly_kj = ln(cosh(y_kj))
  // sa_k = \sum_i a_i*spinStates_ki
  cublas::gemm(theCublasHandle_, 1, knChains, knInputs, kone, kzero, PTR_FROM_THRUST(af_dev_.data()),
    spinStates_dev, PTR_FROM_THRUST(sa_dev_.data()));
  // lnpsi_k = \sum_j ly_kj + sa_k
  CHECK_ERROR(cudaSuccess, cudaMemcpy(lnpsi_dev, PTR_FROM_THRUST(sa_dev_.data()),
    sizeof(thrust::complex<FloatType>)*knChains, cudaMemcpyDeviceToDevice));
  cublas::gemm(theCublasHandle_, 1, knChains, kAlpha*knInputs, kone, kone, PTR_FROM_THRUST(koneHiddens_dev.data()),
    PTR_FROM_THRUST(ly_dev_.data()), lnpsi_dev);
  if (saveSpinStates)
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(spinStates_dev_.data()), spinStates_dev,
      sizeof(thrust::complex<FloatType>)*spinStates_dev_.size(), cudaMemcpyDeviceToDevice));
}

template <typename FloatType>
void RBMTrSymm<FloatType>::backward(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  gpu_kernel::RBMTrSymm__GetGradientsOfParameters__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, kAlpha, knChains,
    PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), d_dw_dev_, d_da_dev_, d_db_dev_);
  CHECK_ERROR(cudaSuccess, cudaMemcpy(lnpsiGradients_dev, PTR_FROM_THRUST(lnpsiGradients_dev_.data()),
    sizeof(std::complex<FloatType>)*variables_dev_.size()*knChains, cudaMemcpyDeviceToDevice));
}

template <typename FloatType>
void RBMTrSymm<FloatType>::update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate)
{
  gpu_kernel::update_parameters<<<kgpuBlockSize4, NUM_THREADS_PER_BLOCK>>>(variables_dev_.size(),
    derivativeLoss_dev, learningRate, PTR_FROM_THRUST(variables_dev_.data()));
  this->symmetrize_variables();
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, kAlpha*knInputs, knChains, kone, PTR_FROM_THRUST(bf_dev_.data()),
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, kAlpha*knInputs, knChains, knInputs, kone, kone,
    PTR_FROM_THRUST(wf_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  // sa_k = \sum_i a_i*spinStates_ki
  cublas::gemm(theCublasHandle_, 1, knChains, knInputs, kone, kzero, PTR_FROM_THRUST(af_dev_.data()),
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(sa_dev_.data()));
}

template <typename FloatType>
void RBMTrSymm<FloatType>::spin_flip(const bool * isSpinFlipped_dev, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  gpu_kernel::conditional_y_update<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, kAlpha*knInputs, knChains, index_,
    isSpinFlipped_dev, PTR_FROM_THRUST(wf_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::RBM__saUpdate__<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, index_,
    isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(af_dev_.data()), PTR_FROM_THRUST(sa_dev_.data()));
  gpu_kernel::conditional_spin_update<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, index_,
    isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()));
}

template <typename FloatType>
void RBMTrSymm<FloatType>::save(const std::string filePath, const int precision)
{
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  variables_host_ = variables_dev_ ;
  for (const auto & var : variables_host_)
    writer << var << " ";
  writer.close();
}

template <typename FloatType>
void RBMTrSymm<FloatType>::load(const std::string filePath)
{
  // read rawdata from the text file located at 'filePath'
  std::vector<std::complex<FloatType>> rawdata;
  std::ifstream reader(filePath);
  if (reader.is_open())
  {
    std::complex<FloatType> temp;
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
  if (rawdata.size() == variables_host_.size())
  {
    for (int i=0; i<variables_host_.size(); ++i)
      variables_host_[i] = rawdata[i];
    variables_dev_ = variables_host_;
  }
  else
    std::cout << " check parameter size... " << std::endl;
}

template <typename FloatType>
void RBMTrSymm<FloatType>::copy_to(RBMTrSymm<FloatType> & rbm) const
{
  if (knChains != rbm.get_nChains())
    throw std::length_error("knChains != rbm.get_nChains()");
  if (knInputs != rbm.get_nInputs())
    throw std::length_error("knInputs != rbm.get_nInputs()");
  if (kAlpha != rbm.get_alpha())
    throw std::length_error("kAlpha != rm.get_alpha()");
  rbm.variables_dev_ = variables_dev_;
  rbm.symmetrize_variables();
}

template <typename FloatType>
void RBMTrSymm<FloatType>::symmetrize_variables()
{
  gpu_kernel::RBMTrSymm__ConstructWeightAndBias__<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(kAlpha, knInputs,
    w_dev_, a_dev_, b_dev_, PTR_FROM_THRUST(wf_dev_.data()), PTR_FROM_THRUST(af_dev_.data()), PTR_FROM_THRUST(bf_dev_.data()));
}


template <typename FloatType>
FFNN<FloatType>::FFNN(const int nInputs, const int nHiddens, const int nChains):
  knInputs(nInputs),
  knHiddens(nHiddens),
  knChains(nChains),
  variables_host_(nInputs*nHiddens + 2*nHiddens),
  variables_dev_(nInputs*nHiddens + 2*nHiddens),
  lnpsiGradients_dev_(nChains*(nInputs*nHiddens + 2*nHiddens)),
  spinStates_dev_(nInputs*nChains),
  y_dev_(nHiddens*nChains),
  acty_dev_(nHiddens*nChains),
  index_(0),
  kzero(0.0, 0.0),
  kone(1.0, 0.0),
  koneChains_dev(nChains, thrust::complex<FloatType>(1.0, 0.0)),
  kgpuBlockSize1(CHECK_BLOCK_SIZE(1+(nChains*nHiddens-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize2(CHECK_BLOCK_SIZE(1+(nInputs*nHiddens-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize3(CHECK_BLOCK_SIZE(1+(nChains-1)/NUM_THREADS_PER_BLOCK))
{
  // parameter initialization: starting from the Gaussian random distribution
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double> randwi1(0, std::sqrt(1.0/(nInputs+nHiddens))), randw1o(0, std::sqrt(1.0/nHiddens));
  // host
  wi1_host_ = &variables_host_[0];
  b1_host_ = &variables_host_[nInputs*nHiddens];
  w1o_host_ = &variables_host_[nInputs*nHiddens + nHiddens];
  // device
  wi1_dev_ = PTR_FROM_THRUST(&variables_dev_[0]);
  b1_dev_ = PTR_FROM_THRUST(&variables_dev_[nInputs*nHiddens]);
  w1o_dev_ = PTR_FROM_THRUST(&variables_dev_[nInputs*nHiddens + nHiddens]);
  for (int i=0; i<nInputs*nHiddens; ++i)
    wi1_host_[i] = thrust::complex<FloatType>(randwi1(ran), 1e-1*randwi1(ran));
  for (int j=0; j<nHiddens; ++j)
    b1_host_[j] = 0.0;
  for (int j=0; j<nHiddens; ++j)
    w1o_host_[j] = thrust::complex<FloatType>(randw1o(ran), 1e-1*randw1o(ran));
  variables_dev_ = variables_host_; // copy memory from host to device
  d_dwi1_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[0]);
  d_db1_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[nInputs*nHiddens]);
  d_dw1o_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[nInputs*nHiddens + nHiddens]);
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
  const float ln2f = std::log(2.0f);
  const double ln2d = std::log(2.0);
  CHECK_ERROR(cudaSuccess, cudaMemcpyToSymbol(gpu_device::kln2f, &ln2f, sizeof(float)));
  CHECK_ERROR(cudaSuccess, cudaMemcpyToSymbol(gpu_device::kln2d, &ln2d, sizeof(double)));
}

template <typename FloatType>
FFNN<FloatType>::~FFNN()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename FloatType>
void FFNN<FloatType>::initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev)
{
  if (spinStates_dev == nullptr)
    thrust::fill(spinStates_dev_.begin(), spinStates_dev_.end(), kone); // spin states are initialized with 1
  else
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(spinStates_dev_.data()), spinStates_dev,
      sizeof(thrust::complex<FloatType>)*spinStates_dev_.size(), cudaMemcpyDeviceToDevice));
  // y_kj = \sum_i spinStates_ki wi1_ij + koneChains_k (x) b1_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, knHiddens, knChains, kone, b1_dev_,
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, knHiddens, knChains, knInputs, kone, kone,
    wi1_dev_, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(y_dev_.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(acty_dev_.data()));  // acty_kj = ln(cosh(y_kj))
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kzero, w1o_dev_,
    PTR_FROM_THRUST(acty_dev_.data()), lnpsi_dev); // lnpsi_k = \sum_j acty_kj w1o_j
}

template <typename FloatType>
void FFNN<FloatType>::forward(const int spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev)
{
  index_ = spinFlipIndex;
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains, index_, wi1_dev_,
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(acty_dev_.data()));
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kzero, w1o_dev_, PTR_FROM_THRUST(acty_dev_.data()), lnpsi_dev);
}

template <typename FloatType>
void FFNN<FloatType>::forward(const thrust::complex<FloatType> * spinStates_dev, thrust::complex<FloatType> * lnpsi_dev, const bool saveSpinStates)
{
  // y_kj = \sum_i spinStates_ki wi1_ij + koneChains_k (x) b1_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, knHiddens, knChains, kone, b1_dev_,
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, knHiddens, knChains, knInputs, kone, kone,
    wi1_dev_, spinStates_dev, PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(y_dev_.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(acty_dev_.data()));  // acty_kj = ln(cosh(y_kj))
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kzero, w1o_dev_,
    PTR_FROM_THRUST(acty_dev_.data()), lnpsi_dev); // lnpsi_k = \sum_j acty_kj w1o_j
  if (saveSpinStates)
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(spinStates_dev_.data()), spinStates_dev,
      sizeof(thrust::complex<FloatType>)*spinStates_dev_.size(), cudaMemcpyDeviceToDevice));
}

template <typename FloatType>
void FFNN<FloatType>::forward(const thrust::device_vector<thrust::pair<int, int> > & spinPairFlipIdx_dev,
  thrust::complex<FloatType> * lnpsi_dev)
{
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains,
    PTR_FROM_THRUST(spinPairFlipIdx_dev.data()), wi1_dev_, PTR_FROM_THRUST(spinStates_dev_.data()),
    PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(acty_dev_.data()));
  cublas::gemm(theCublasHandle_, 1, knChains, knHiddens, kone, kzero, w1o_dev_, PTR_FROM_THRUST(acty_dev_.data()), lnpsi_dev);
}

template <typename FloatType>
void FFNN<FloatType>::backward(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  gpu_kernel::FFNN__GetGradientsOfParameters__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains,
    PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), w1o_dev_, d_dwi1_dev_, d_db1_dev_, d_dw1o_dev_);
  gpu_kernel::FFNN__GetlnpsiGradients__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs,
    knHiddens, knChains, d_dwi1_dev_, d_db1_dev_, d_dw1o_dev_, lnpsiGradients_dev);
}

template <typename FloatType>
void FFNN<FloatType>::update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate)
{
  gpu_kernel::FFNN__UpdateParameters__<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(knInputs,
    knHiddens, knChains, derivativeLoss_dev, learningRate, wi1_dev_, b1_dev_, w1o_dev_);
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, knHiddens, knChains, kone, b1_dev_,
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, knHiddens, knChains, knInputs, kone, kone,
    wi1_dev_, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
}

template <typename FloatType>
void FFNN<FloatType>::spin_flip(const bool * isSpinFlipped_dev, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  gpu_kernel::conditional_y_update<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains, index_,
    isSpinFlipped_dev, wi1_dev_, PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::conditional_spin_update<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, index_,
    isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()));
}

template <typename FloatType>
void FFNN<FloatType>::spin_flip(const bool * isSpinFlipped_dev, const thrust::device_vector<thrust::pair<int, int> > & spinPairFlipIdx_dev)
{
  gpu_kernel::conditional_y_update<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, knHiddens, knChains,
    PTR_FROM_THRUST(spinPairFlipIdx_dev.data()), isSpinFlipped_dev, wi1_dev_, PTR_FROM_THRUST(spinStates_dev_.data()),
    PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::conditional_spin_update<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains,
    PTR_FROM_THRUST(spinPairFlipIdx_dev.data()), isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()));
}

template <typename FloatType>
void FFNN<FloatType>::save(const FFNNDataType typeInfo, const std::string filePath, const int precision, const bool useCopyFromDeviceToHost)
{
  if (useCopyFromDeviceToHost)
    variables_host_ = variables_dev_; // copy memory from device to host
  // save data according to the FFNNDataType
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  if (typeInfo == FFNNDataType::W1)
  {
    for (int i=0; i<knInputs; ++i)
    {
      for (int j=0; j<knHiddens; ++j)
        writer << wi1_host_[i*knHiddens+j] << " ";
      writer << std::endl;
    }
  }
  else if (typeInfo == FFNNDataType::W2)
  {
    for (int j=0; j<knHiddens; ++j)
      writer << w1o_host_[j] << " ";
    writer << std::endl;
  }
  else if (typeInfo == FFNNDataType::B1)
    for (int j=0; j<knHiddens; ++j)
      writer << b1_host_[j] << " ";
  writer.close();
}

template <typename FloatType>
void FFNN<FloatType>::save(const std::string prefix, const int precision)
{
  variables_host_ = variables_dev_; // copy memory from device to host
  this->save(FFNNDataType::W1, prefix + "Dw1.dat", precision, false);
  this->save(FFNNDataType::W2, prefix + "Dw2.dat", precision, false);
  this->save(FFNNDataType::B1, prefix + "Db1.dat", precision, false);
}

template <typename FloatType>
void FFNN<FloatType>::load(const FFNNDataType typeInfo, const std::string filePath)
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
  if (typeInfo == FFNNDataType::W1)
  {
    if (rawdata.size() == knInputs*knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        wi1_host_[i] = rawdata[i];
    else
      std::cout << "# check 'w1' size... " << std::endl;
  }
  else if (typeInfo == FFNNDataType::W2)
  {
    if (rawdata.size() == knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        w1o_host_[i] = rawdata[i];
    else
      std::cout << "# check 'w2' size... " << std::endl;
  }
  else if (typeInfo == FFNNDataType::B1)
  {
    if (rawdata.size() == knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        b1_host_[i] = rawdata[i];
    else
      std::cout << "# check 'b1' size... " << std::endl;
  }
  variables_dev_ = variables_host_; // copy memory from host to device
}

template <typename FloatType>
void FFNN<FloatType>::load(const std::string prefix)
{
  this->load(FFNNDataType::W1, prefix + "Dw1.dat");
  this->load(FFNNDataType::W2, prefix + "Dw2.dat");
  this->load(FFNNDataType::B1, prefix + "Db1.dat");
}

template <typename FloatType>
void FFNN<FloatType>::copy_to(FFNN<FloatType> & fnn) const
{
  if (knChains != fnn.get_nChains())
    throw std::length_error("knChains != fnn.get_nChains()");
  if (knInputs != fnn.get_nInputs())
    throw std::length_error("knInputs != fnn.get_nInputs()");
  if (knHiddens != fnn.get_nHiddens())
    throw std::length_error("knHiddens != fnn.get_nHiddens()");
  fnn.variables_dev_ = variables_dev_;
}

template <typename FloatType>
void FFNN<FloatType>::look_inside() const
{
  std::cout << "---- wi1" << std::flush << std::endl;
  gpu_kernel::common__Print__<<<1,1>>>(wi1_dev_, knInputs, knHiddens);
  cudaDeviceSynchronize();
  std::cout << "---- b1" << std::flush << std::endl;
  gpu_kernel::common__Print__<<<1,1>>>(b1_dev_, 1, knHiddens);
  cudaDeviceSynchronize();
  std::cout << "---- w1o" << std::flush << std::endl;
  gpu_kernel::common__Print__<<<1,1>>>(w1o_dev_, 1, knHiddens);
  cudaDeviceSynchronize();
}


template <typename FloatType>
FFNNTrSymm<FloatType>::FFNNTrSymm(const int nInputs, const int alpha, const int nChains):
  knInputs(nInputs),
  kAlpha(alpha),
  knChains(nChains),
  variables_host_(nInputs*alpha+2*alpha),
  variables_dev_(nInputs*alpha+2*alpha),
  lnpsiGradients_dev_(nChains*(nInputs*alpha+2*alpha)),
  spinStates_dev_(nInputs*nChains),
  y_dev_(alpha*nInputs*nChains),
  acty_dev_(alpha*nInputs*nChains),
  wi1f_dev_(nInputs*alpha*nInputs),
  b1f_dev_(alpha*nInputs),
  w1of_dev_(alpha*nInputs),
  index_(0),
  kzero(0.0, 0.0),
  kone(1.0, 0.0),
  koneChains_dev(nChains, thrust::complex<FloatType>(1.0, 0.0)),
  kgpuBlockSize1(CHECK_BLOCK_SIZE(1+(nChains*alpha*nInputs-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize2(CHECK_BLOCK_SIZE(1+(nInputs*alpha*nInputs-1)/NUM_THREADS_PER_BLOCK)),
  kgpuBlockSize3(CHECK_BLOCK_SIZE(1+(nChains-1)/NUM_THREADS_PER_BLOCK))
{
  // parameter initialization: starting from the Gaussian random distribution
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double>
    randwi1(0, std::sqrt(1.0/((1+kAlpha)*knInputs))),
    randw1o(0, std::sqrt(1.0/(kAlpha*knInputs)));
  // host
  wi1_host_ = &variables_host_[0];
  b1_host_ = &variables_host_[knInputs*kAlpha];
  w1o_host_ = &variables_host_[knInputs*kAlpha+kAlpha];
  // device
  wi1_dev_ = PTR_FROM_THRUST(&variables_dev_[0]);
  b1_dev_ = PTR_FROM_THRUST(&variables_dev_[knInputs*kAlpha]);
  w1o_dev_ = PTR_FROM_THRUST(&variables_dev_[knInputs*kAlpha+kAlpha]);
  for (int i=0; i<knInputs*kAlpha; ++i)
    wi1_host_[i] = thrust::complex<FloatType>(randwi1(ran), 1e-1*randwi1(ran));
  for (int j=0; j<kAlpha; ++j)
    b1_host_[j] = 0.0;
  for (int j=0; j<kAlpha; ++j)
    w1o_host_[j] = thrust::complex<FloatType>(randw1o(ran), 1e-1*randw1o(ran));
  variables_dev_ = variables_host_; // copy memory from host to device
  d_dwi1_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[0]);
  d_db1_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[knInputs*kAlpha]);
  d_dw1o_dev_ = PTR_FROM_THRUST(&lnpsiGradients_dev_[knInputs*kAlpha+kAlpha]);
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasCreate(&theCublasHandle_)); // create cublas handler
  const float ln2f = std::log(2.0f);
  const double ln2d = std::log(2.0);
  CHECK_ERROR(cudaSuccess, cudaMemcpyToSymbol(gpu_device::kln2f, &ln2f, sizeof(float)));
  CHECK_ERROR(cudaSuccess, cudaMemcpyToSymbol(gpu_device::kln2d, &ln2d, sizeof(double)));
}

template <typename FloatType>
FFNNTrSymm<FloatType>::~FFNNTrSymm()
{
  CHECK_ERROR(CUBLAS_STATUS_SUCCESS, cublasDestroy(theCublasHandle_));
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::initialize(thrust::complex<FloatType> * lnpsi_dev, const thrust::complex<FloatType> * spinStates_dev)
{
  if (spinStates_dev == nullptr)
    thrust::fill(spinStates_dev_.begin(), spinStates_dev_.end(), kone); // spin states are initialized with 1
  else
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(spinStates_dev_.data()), spinStates_dev,
      sizeof(thrust::complex<FloatType>)*spinStates_dev_.size(), cudaMemcpyDeviceToDevice));
  // construct full weight matrices and a bias vector
  this->symmetrize_variables();
  // y_kj = \sum_i spinStates_ki wi1_ij + koneChains_k (x) b1_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, kAlpha*knInputs, knChains, kone, PTR_FROM_THRUST(b1f_dev_.data()),
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, kAlpha*knInputs, knChains, knInputs, kone, kone,
    PTR_FROM_THRUST(wi1f_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(y_dev_.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(acty_dev_.data()));  // acty_kj = ln(cosh(y_kj))
  cublas::gemm(theCublasHandle_, 1, knChains, kAlpha*knInputs, kone, kzero, PTR_FROM_THRUST(w1of_dev_.data()),
    PTR_FROM_THRUST(acty_dev_.data()), lnpsi_dev); // lnpsi_k = \sum_j acty_kj w1o_j
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::forward(const int spinFlipIndex, thrust::complex<FloatType> * lnpsi_dev)
{
  index_ = spinFlipIndex;
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, kAlpha*knInputs, knChains, index_, PTR_FROM_THRUST(wi1f_dev_.data()),
    PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(acty_dev_.data()));
  cublas::gemm(theCublasHandle_, 1, knChains, kAlpha*knInputs, kone, kzero, PTR_FROM_THRUST(w1of_dev_.data()),
    PTR_FROM_THRUST(acty_dev_.data()), lnpsi_dev);
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::forward(const thrust::complex<FloatType> * spinStates_dev, thrust::complex<FloatType> * lnpsi_dev, const bool saveSpinStates)
{
  // y_kj = \sum_i spinStates_ki wi1_ij + koneChains_k (x) b1_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, kAlpha*knInputs, knChains, kone, PTR_FROM_THRUST(b1f_dev_.data()),
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, kAlpha*knInputs, knChains, knInputs, kone, kone,
    PTR_FROM_THRUST(wi1f_dev_.data()), spinStates_dev, PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::logcosh<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(y_dev_.size(), PTR_FROM_THRUST(y_dev_.data()),
    PTR_FROM_THRUST(acty_dev_.data()));  // acty_kj = ln(cosh(y_kj))
  cublas::gemm(theCublasHandle_, 1, knChains, kAlpha*knInputs, kone, kzero, PTR_FROM_THRUST(w1of_dev_.data()),
    PTR_FROM_THRUST(acty_dev_.data()), lnpsi_dev); // lnpsi_k = \sum_j acty_kj w1o_j
  if (saveSpinStates)
    CHECK_ERROR(cudaSuccess, cudaMemcpy(PTR_FROM_THRUST(spinStates_dev_.data()), spinStates_dev,
      sizeof(thrust::complex<FloatType>)*spinStates_dev_.size(), cudaMemcpyDeviceToDevice));
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::backward(thrust::complex<FloatType> * lnpsiGradients_dev)
{
  gpu_kernel::FFNNTrSymm__GetGradientsOfParameters__<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs,
    kAlpha, knChains, PTR_FROM_THRUST(y_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()),
    PTR_FROM_THRUST(w1of_dev_.data()), d_dwi1_dev_, d_db1_dev_, d_dw1o_dev_);
  CHECK_ERROR(cudaSuccess, cudaMemcpy(lnpsiGradients_dev, PTR_FROM_THRUST(lnpsiGradients_dev_.data()),
    sizeof(std::complex<FloatType>)*variables_dev_.size()*knChains, cudaMemcpyDeviceToDevice));
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::update_variables(const thrust::complex<FloatType> * derivativeLoss_dev, const FloatType learningRate)
{
  gpu_kernel::update_parameters<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(variables_dev_.size(),
    derivativeLoss_dev, learningRate, PTR_FROM_THRUST(variables_dev_.data()));
  // construct full weight matrices and a bias vector
  this->symmetrize_variables();
  // y_kj = \sum_i spinStates_ki wi1_ij + koneChains_k (x) b1_j
  thrust::fill(y_dev_.begin(), y_dev_.end(), kzero);
  cublas::ger(theCublasHandle_, kAlpha*knInputs, knChains, kone, PTR_FROM_THRUST(b1f_dev_.data()),
    PTR_FROM_THRUST(koneChains_dev.data()), PTR_FROM_THRUST(y_dev_.data()));
  cublas::gemm(theCublasHandle_, kAlpha*knInputs, knChains, knInputs, kone, kone,
    PTR_FROM_THRUST(wi1f_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::spin_flip(const bool * isSpinFlipped_dev, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  gpu_kernel::conditional_y_update<<<kgpuBlockSize1, NUM_THREADS_PER_BLOCK>>>(knInputs, kAlpha*knInputs, knChains, index_,
    isSpinFlipped_dev, PTR_FROM_THRUST(wi1f_dev_.data()), PTR_FROM_THRUST(spinStates_dev_.data()), PTR_FROM_THRUST(y_dev_.data()));
  gpu_kernel::conditional_spin_update<<<kgpuBlockSize3, NUM_THREADS_PER_BLOCK>>>(knInputs, knChains, index_,
    isSpinFlipped_dev, PTR_FROM_THRUST(spinStates_dev_.data()));
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::save(const std::string filePath, const int precision)
{
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  variables_host_ = variables_dev_ ;
  for (const auto & var : variables_host_)
    writer << var << " ";
  writer.close();
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::load(const std::string filePath)
{
  // read rawdata from the text file located at 'filePath'
  std::vector<std::complex<FloatType>> rawdata;
  std::ifstream reader(filePath);
  if (reader.is_open())
  {
    std::complex<FloatType> temp;
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
  if (rawdata.size() == variables_host_.size())
  {
    for (int i=0; i<variables_host_.size(); ++i)
      variables_host_[i] = rawdata[i];
    variables_dev_ = variables_host_;
  }
  else
    std::cout << " check parameter size... " << std::endl;
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::copy_to(FFNNTrSymm<FloatType> & fnn) const
{
  if (knChains != fnn.get_nChains())
    throw std::length_error("knChains != fnn.get_nChains()");
  if (knInputs != fnn.get_nInputs())
    throw std::length_error("knInputs != fnn.get_nInputs()");
  if (kAlpha != fnn.get_alpha())
    throw std::length_error("kAlpha != fnn.get_alpha()");
  fnn.variables_dev_ = variables_dev_;
  fnn.symmetrize_variables();
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::symmetrize_variables()
{
  gpu_kernel::FFNNTrSymm__ConstructWeightAndBias__<<<kgpuBlockSize2, NUM_THREADS_PER_BLOCK>>>(kAlpha,
    knInputs, wi1_dev_, b1_dev_, w1o_dev_, PTR_FROM_THRUST(wi1f_dev_.data()),
    PTR_FROM_THRUST(b1f_dev_.data()), PTR_FROM_THRUST(w1of_dev_.data()));
}
} // end namespace spinhalf

namespace gpu_device
{
__device__ thrust::complex<float> logcosh(const thrust::complex<float> z)
{
  const float x = z.real(), y = z.imag();
  const float absx = abs(x), cosy = cosf(y), siny = sinf(y);
  const float expabsm2x = expf(-2.0f*absx);
  const float real = (1.0f+expabsm2x)*cosy, imag = (1.0f-expabsm2x)*siny*copysign(1.0f, x);
  return thrust::log(thrust::complex<float>(real, imag))+(absx-kln2f[0]);
}

__device__ thrust::complex<double> logcosh(const thrust::complex<double> z)
{
  const double x = z.real(), y = z.imag();
  const double absx = abs(x), cosy = cos(y), siny = sin(y);
  const double expabsm2x = exp(-2.0*absx);
  const double real = (1.0+expabsm2x)*cosy, imag = (1.0-expabsm2x)*siny*copysign(1.0, x);
  return thrust::log(thrust::complex<double>(real, imag))+(absx-kln2d[0]);
}
}


namespace gpu_kernel
{
// GPU kernels for common uses
template <typename FloatType>
__global__ void logcosh(const int size, const thrust::complex<FloatType> * y, thrust::complex<FloatType> * z)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < size)
  {
    z[idx] = gpu_device::logcosh(y[idx]);
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void logcosh(const int nInputs, const int nHiddens, const int nChains, const int spinFlipIndex,
  const thrust::complex<FloatType> * wi1, const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * y, thrust::complex<FloatType> * z)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType two = 2;
  while (idx < nChains*nHiddens)
  {
    const int k = idx/nHiddens, j = idx-nHiddens*k;
    z[idx] = gpu_device::logcosh(y[idx]-wi1[spinFlipIndex*nHiddens+j]*(two*spinStates[k*nInputs+spinFlipIndex].real()));
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void logcosh(const int nInputs, const int nHiddens, const int nChains,
  const thrust::pair<int, int> * spinPairFlipIdx,
  const thrust::complex<FloatType> * wi1, const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * y, thrust::complex<FloatType> * z)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType two = 2;
  while (idx < nChains*nHiddens)
  {
    const int k = idx/nHiddens, j = idx-nHiddens*k,
      spinFlipIdx1 = spinPairFlipIdx[k].first, spinFlipIdx2 = spinPairFlipIdx[k].second;
    z[idx] = gpu_device::logcosh(y[idx]
      -wi1[spinFlipIdx1*nHiddens+j]*(two*spinStates[k*nInputs+spinFlipIdx1].real())
      -wi1[spinFlipIdx2*nHiddens+j]*(two*spinStates[k*nInputs+spinFlipIdx2].real()));
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void update_parameters(const int size, const thrust::complex<FloatType> * derivativeLoss,
  const FloatType learningRate, thrust::complex<FloatType> * variables)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x; 
  // update wi1
  while (idx < size)
  {
    variables[idx] = variables[idx]-learningRate*derivativeLoss[idx];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void conditional_y_update(const int nInputs, const int nHiddens,
  const int nChains, const int spinFlipIndex, const bool * isSpinFlipped,
  const thrust::complex<FloatType> * w, thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * y)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType twoDelta[2] = {0.0, 2.0};
  // update y: y_{k,j} = y_{k,j} - 2*delta(k)*wi1_{spinFlipIndex,j}*spinStates_{k,spinFlipIndex}
  while (idx < nChains*nHiddens)
  {
    const int k = idx/nHiddens, j = idx-nHiddens*k;
    y[idx] = y[idx]-w[spinFlipIndex*nHiddens+j]*(twoDelta[isSpinFlipped[k]]*spinStates[k*nInputs+spinFlipIndex].real());
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void conditional_y_update(const int nInputs, const int nHiddens,
  const int nChains, const thrust::pair<int, int> * spinPairFlipIdx, const bool * isSpinFlipped,
  const thrust::complex<FloatType> * w, thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * y)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType twoDelta[2] = {0.0, 2.0};
  // update y: y_{k,j} = y_{k,j} - 2*delta(k)*wi1_{spinFlipIndex,j}*spinStates_{k,spinFlipIndex}
  while (idx < nChains*nHiddens)
  {
    const int k = idx/nHiddens,
              j = idx-nHiddens*k;
    const int spinFlipIdx1 = spinPairFlipIdx[k].first,
              spinFlipIdx2 = spinPairFlipIdx[k].second;
    y[idx] = y[idx]-twoDelta[isSpinFlipped[k]]*
      (w[spinFlipIdx1*nHiddens+j]*spinStates[k*nInputs+spinFlipIdx1].real()
      +w[spinFlipIdx2*nHiddens+j]*spinStates[k*nInputs+spinFlipIdx2].real());
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void conditional_spin_update(const int nInputs, const int nChains, const int spinFlipIndex,
  const bool * isSpinFlipped, thrust::complex<FloatType> * spinStates)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType one = 1.0, twoDelta[2] = {0.0, 2.0};
  // update spin: spinState_{k,spinFlipIndex} = (1-2*delta(k))*spinStates_{k,spinFlipIndex}
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    const int k = idx;
    spinStates[k*nInputs+spinFlipIndex] = (one-twoDelta[isSpinFlipped[k]])*spinStates[k*nInputs+spinFlipIndex].real();
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void conditional_spin_update(const int nInputs, const int nChains, const thrust::pair<int, int> * spinPairFlipIdx,
  const bool * isSpinFlipped, thrust::complex<FloatType> * spinStates)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType one = 1.0, twoDelta[2] = {0.0, 2.0};
  // update spin: spinState_{k,spinFlipIndex} = (1-2*delta(k))*spinStates_{k,spinFlipIndex}
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    const int k = idx;
    const int spinFlipIdx1 = spinPairFlipIdx[k].first,
              spinFlipIdx2 = spinPairFlipIdx[k].second;
    spinStates[k*nInputs+spinFlipIdx1] = (one-twoDelta[isSpinFlipped[k]])*spinStates[k*nInputs+spinFlipIdx1].real();
    spinStates[k*nInputs+spinFlipIdx2] = (one-twoDelta[isSpinFlipped[k]])*spinStates[k*nInputs+spinFlipIdx2].real();
    idx += nstep;
  }
}

// GPU kernels for RBM and RBMTrSymm
template <typename FloatType>
__global__ void RBM__sadot__(const int nInputs, const int nChains, const int spinFlipIndex,
  const thrust::complex<FloatType> * sa, const thrust::complex<FloatType> * a,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * lnpsi)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x; 
  const FloatType two = 2;
  while (idx < nChains)
  {
    lnpsi[idx] = sa[idx]-(two*spinStates[idx*nInputs+spinFlipIndex].real())*a[spinFlipIndex];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void RBM__sadot__(const int nInputs, const int nChains, const thrust::pair<int, int> * spinPairFlipIdx,
  const thrust::complex<FloatType> * sa, const thrust::complex<FloatType> * a,
  const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * lnpsi)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x; 
  const FloatType two = 2;
  while (idx < nChains)
  {
    const int k = idx;
    const int spinFlipIdx1 = spinPairFlipIdx[k].first,
              spinFlipIdx2 = spinPairFlipIdx[k].second;
    lnpsi[k] = sa[k]
      -(two*spinStates[k*nInputs+spinFlipIdx1].real())*a[spinFlipIdx1]
      -(two*spinStates[k*nInputs+spinFlipIdx2].real())*a[spinFlipIdx2];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void RBM__GetGradientsOfParameters__(const int nInputs, const int nHiddens, const int nChains,
  const thrust::complex<FloatType> * y, const thrust::complex<FloatType> * spinStates,
  thrust::complex<FloatType> * d_dw, thrust::complex<FloatType> * d_da, thrust::complex<FloatType> * d_db)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x; 
  const int vSize = nInputs*nHiddens+nInputs+nHiddens;
  while (idx < nChains*nInputs)
  {
    const int k = idx/nInputs, i = idx-k*nInputs;
    for (int j=0; j<nHiddens; ++j)
      d_dw[k*vSize+i*nHiddens+j] = spinStates[k*nInputs+i].real()*thrust::tanh(y[k*nHiddens+j]);
    d_da[k*vSize+i] = spinStates[k*nInputs+i];
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains*nHiddens)
  {
    const int k = idx/nHiddens, j = idx-k*nHiddens;
    d_db[k*vSize+j] = thrust::tanh(y[k*nHiddens+j]);
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void RBM__saUpdate__(const int nInputs, const int nChains, const int spinFlipIndex,
  const bool * isSpinFlipped, const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * a, thrust::complex<FloatType> * sa)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType twoDelta[2] = {0.0, 2.0};
  // update sa: sa_{k} = sa_{k} - 2*delta(k)*a_{spinFlipIndex}*spinStates_{k,spinFlipIndex}
  while (idx < nChains)
  {
    sa[idx] = sa[idx]-(twoDelta[isSpinFlipped[idx]]*spinStates[idx*nInputs+spinFlipIndex].real())*a[spinFlipIndex];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void RBM__saUpdate__(const int nInputs, const int nChains, const thrust::pair<int, int> * spinPairFlipIdx,
  const bool * isSpinFlipped, const thrust::complex<FloatType> * spinStates,
  const thrust::complex<FloatType> * a, thrust::complex<FloatType> * sa)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const FloatType twoDelta[2] = {0.0, 2.0};
  // update sa: sa_{k} = sa_{k} - 2*delta(k)*a_{spinFlipIndex}*spinStates_{k,spinFlipIndex}
  while (idx < nChains)
  {
    const int k = idx;
    const int spinFlipIdx1 = spinPairFlipIdx[k].first,
              spinFlipIdx2 = spinPairFlipIdx[k].second;
    sa[k] = sa[k]-twoDelta[isSpinFlipped[k]]*(spinStates[k*nInputs+spinFlipIdx1].real()*a[spinFlipIdx1]
                                             +spinStates[k*nInputs+spinFlipIdx2].real()*a[spinFlipIdx2]);
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void RBMTrSymm__GetGradientsOfParameters__(const int nInputs, const int alpha, const int nChains,
  const thrust::complex<FloatType> * y, const thrust::complex<FloatType> * spinStates, thrust::complex<FloatType> * d_dw,
  thrust::complex<FloatType> * d_da, thrust::complex<FloatType> * d_db)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned int vSize = nInputs*alpha+alpha+1;
  const FloatType zero = 0;
  while (idx < (nChains*alpha*nInputs))
  {
    const int k = idx/(alpha*nInputs), f = (idx-k*(alpha*nInputs))/nInputs, i = idx-k*(alpha*nInputs)-f*nInputs;
    d_dw[k*vSize+f*nInputs+i] = zero;
    for (int j=0; j<nInputs; ++j)
      d_dw[k*vSize+f*nInputs+i] += thrust::tanh(y[k*alpha*nInputs+f*nInputs+j])*spinStates[k*nInputs+(nInputs+i-j)%nInputs].real();
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains*alpha)
  {
    const int k = idx/alpha, f = idx-k*alpha;
    d_db[k*vSize+f] = zero;
    for (int j=0; j<nInputs; ++j)
      d_db[k*vSize+f] += thrust::tanh(y[k*alpha*nInputs+f*nInputs+j]);
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    d_da[idx*vSize+0] = zero;
    for (int i=0; i<nInputs; ++i)
      d_da[idx*vSize+0] += spinStates[idx*nInputs+i].real();
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void RBMTrSymm__ConstructWeightAndBias__(const int alpha, const int nInputs,
  const thrust::complex<FloatType> * w, const thrust::complex<FloatType> * a, const thrust::complex<FloatType> * b,
  thrust::complex<FloatType> * wf, thrust::complex<FloatType> * af, thrust::complex<FloatType> * bf)
{
  // i=0,1,...,N-1; j=0,1,...,N-1; f=0,...,alpha-1
  // wf_{i,f*nInputs+j} (=wf_{j,f*nInputs+i}), bf_[f*nInputs+j], af_[j]
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x; 
  while (idx < alpha*nInputs*nInputs)
  {
    // idx = f*nInputs*nInputs + j*nInputs + i
    // f : alpha, 0 < i,j < nInputs
    const unsigned int f = idx/(nInputs*nInputs), j = (idx-f*nInputs*nInputs)/nInputs, i = idx-f*nInputs*nInputs-j*nInputs;
    wf[i*nInputs*alpha+f*nInputs+j] = w[f*nInputs+(i+j)%nInputs];
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x; 
  while (idx < alpha*nInputs)
  {
    const unsigned int f = idx/nInputs, j = idx-f*nInputs;
    bf[f*nInputs+j] = b[f];
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x; 
  while (idx < nInputs)
  {
    af[idx] = a[0];
    idx += nstep;
  }
}


// GPU kernels for FFNN
template <typename FloatType>
__global__ void FFNN__GetGradientsOfParameters__(const int nInputs, const int nHiddens,
  const int nChains, const thrust::complex<FloatType> * y,
  const thrust::complex<FloatType> * spinStates, const thrust::complex<FloatType> * w1o,
  thrust::complex<FloatType> * d_dwi1, thrust::complex<FloatType> * d_db1, thrust::complex<FloatType> * d_dw1o)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  const int varSize = nInputs*nHiddens+2*nHiddens; // # of total variables
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x; // [k,j] : k is the parallel chain number and j is the hidden node index.
  // calculate gradients
  while (idx < nChains*nHiddens)
  {
    const int k = idx/nHiddens, j = idx-k*nHiddens, kv = k*varSize, ki = k*nInputs, kh = k*nHiddens;
    const thrust::complex<FloatType> tany_j = thrust::tanh(y[kh+j]);
    for (int i=0; i<nInputs; ++i)
      d_dwi1[kv+i*nHiddens+j] = tany_j*spinStates[ki+i].real()*w1o[j];
    d_db1[kv+j] = tany_j*w1o[j];
    d_dw1o[kv+j] = gpu_device::logcosh(y[kh+j]);
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void FFNN__GetlnpsiGradients__(const int nInputs, const int nHiddens, const int nChains,
  const thrust::complex<FloatType> * d_dwi1, const thrust::complex<FloatType> * d_db1,
  const thrust::complex<FloatType> * d_dw1o, thrust::complex<FloatType> * lnpsiGradients)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  const int varSize = nInputs*nHiddens+2*nHiddens; // # of total variables
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x; // [k,j] : k is the parallel chain number and j is the hidden node index.
  // save the results into lnpsiGradients
  const int pvarSize = nInputs*nHiddens+2*nHiddens;
  while (idx < nChains*nHiddens)
  {
    const int k = idx/nHiddens, j = idx-k*nHiddens, kv = k*varSize, kpv = k*pvarSize;
    for (int i=0; i<nInputs; ++i)
      lnpsiGradients[kpv+j*nInputs+i] = d_dwi1[kv+i*nHiddens+j];
    lnpsiGradients[kpv+nHiddens*nInputs+j] = d_db1[kv+j];
    lnpsiGradients[kpv+nHiddens*(nInputs+1)+j] = d_dw1o[kv+j];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void FFNN__UpdateParameters__(const int nInputs, const int nHiddens, const int nChains,
  const thrust::complex<FloatType> * derivativeLoss, const FloatType learningRate,
  thrust::complex<FloatType> * wi1, thrust::complex<FloatType> * b1, thrust::complex<FloatType> * w1o)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x; 
  // update wi1
  while (idx < nHiddens*nInputs)
  {
    // idx : [j,i] (j is the hidden node index and i represents the spin site.)
    const int j = idx/nInputs, i = idx-j*nInputs;
    wi1[i*nHiddens+j] = wi1[i*nHiddens+j]-learningRate*derivativeLoss[idx];
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  // update b1 and w1o
  while (idx < nHiddens)
  {
    // idx : [j]
    const int j = idx;
    b1[j] = b1[j]-learningRate*derivativeLoss[nHiddens*nInputs+j];
    w1o[j] = w1o[j]-learningRate*derivativeLoss[nHiddens*(nInputs+1)+j];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void FFNNTrSymm__ConstructWeightAndBias__(const int alpha, const int nInputs,
  const thrust::complex<FloatType> * wi1, const thrust::complex<FloatType> * b1, const thrust::complex<FloatType> * w1o,
  thrust::complex<FloatType> * wi1f, thrust::complex<FloatType> * b1f, thrust::complex<FloatType> * w1of)
{
  // i=0,1,...,N-1; j=0,1,...,N-1; f=0,...,alpha-1
  // wi1f_{i,f*nInputs+j} (=wi1f_{j,f*nInputs+i}), b1f_{f*nInputs+j}, w1of_{f*nInputs+j}
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < alpha*nInputs*nInputs)
  {
    // idx = f*nInputs*nInputs + j*nInputs + i
    // f : alpha, 0 < i,j < nInputs
    const unsigned int f = idx/(nInputs*nInputs), j = (idx-f*nInputs*nInputs)/nInputs, i = idx-f*nInputs*nInputs-j*nInputs;
    wi1f[i*nInputs*alpha+f*nInputs+j] = wi1[f*nInputs+(i+j)%nInputs];
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < alpha*nInputs)
  {
    const unsigned int f = idx/nInputs, j = idx-f*nInputs;
    b1f[f*nInputs+j] = b1[f];
    w1of[f*nInputs+j] = w1o[f];
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void FFNNTrSymm__GetGradientsOfParameters__(const int nInputs, const int alpha, const int nChains,
  const thrust::complex<FloatType> * y, const thrust::complex<FloatType> * spinStates, const thrust::complex<FloatType> * w1of,
  thrust::complex<FloatType> * d_dwi1, thrust::complex<FloatType> * d_db1, thrust::complex<FloatType> * d_dw1o)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const unsigned int vSize = nInputs*alpha+2*alpha;
  const FloatType zero = 0;
  while (idx < nChains*alpha*nInputs)
  {
    const int k = idx/(alpha*nInputs), f = (idx-k*(alpha*nInputs))/nInputs, i = idx-k*(alpha*nInputs)-f*nInputs;
    d_dwi1[k*vSize+f*nInputs+i] = zero;
    for (int j=0; j<nInputs; ++j)
      d_dwi1[k*vSize+f*nInputs+i] += w1of[f*nInputs+j]*thrust::tanh(y[k*alpha*nInputs+f*nInputs+j])*
        spinStates[k*nInputs+(nInputs+i-j)%nInputs].real();
    idx += nstep;
  }
  idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains*alpha)
  {
    const int k = idx/alpha, f = idx-k*alpha;
    d_dw1o[k*vSize+f] = zero;
    d_db1[k*vSize+f] = zero;
    for (int j=0; j<nInputs; ++j)
    {
      d_dw1o[k*vSize+f] += gpu_device::logcosh(y[k*alpha*nInputs+f*nInputs+j]);
      d_db1[k*vSize+f] += w1of[f*nInputs+j]*thrust::tanh(y[k*alpha*nInputs+f*nInputs+j]);
    }
    idx += nstep;
  }
}
} // namespace gpu_kernel
