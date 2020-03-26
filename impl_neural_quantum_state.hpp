// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iostream>
#include <random>
#include <fstream>

template <typename FloatType>
void print(const FloatType * data, const int m, const int n)
{
  for (int i=0; i<m; ++i)
  {
    for (int j=0; j<n; ++j)
      std::cout << data[i*n+j] << " ";
    std::cout << std::endl;
  }
}

namespace spinhalf
{
/*
 1. Index notation
  - i : index for the input layer
  - j :       ""    hidden   ""
  - k :       ""    MCMC chain

 2. member data
  - spin configurations: spinStates_ki
  - variables: w_ij, a_i, b_j
 */
template <typename FloatType>
ComplexRBM<FloatType>::ComplexRBM(const int nInputs, const int nHiddens, const int nChains):
  knInputs(nInputs), knHiddens(nHiddens), knChains(nChains),
  variables_(nInputs*nHiddens + nInputs + nHiddens),
  lnpsiGradients_(nChains*(nInputs*nHiddens + nInputs + nHiddens)),
  spinStates_(nInputs*nChains),
  y_(nHiddens*nChains),
  ly_(nHiddens*nChains),
  sa_(nChains),
  kzero(std::complex<FloatType>(0.0, 0.0)),
  kone(std::complex<FloatType>(1.0, 0.0)),
  ktwo(2.0),
  koneChains(nChains, std::complex<FloatType>(1.0,0.0)),
  koneHiddens(nHiddens, std::complex<FloatType>(1.0,0.0)),
  ktwoTrueFalse({0.0, 2.0}),
  index_(0)
{
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double> randw(0, std::sqrt(1.0/(nInputs+nHiddens))),
                                   randa(0, std::sqrt(1.0/nInputs)),
                                   randb(0, std::sqrt(1.0/nHiddens));
  w_ = &variables_[0],
  a_ = &variables_[knInputs*knHiddens],
  b_ = &variables_[knInputs*knHiddens+nInputs],
  d_dw_ = &lnpsiGradients_[0],
  d_da_ = &lnpsiGradients_[knInputs*knHiddens],
  d_db_ = &lnpsiGradients_[knInputs*knHiddens+nInputs];
  for (int n=0; n<nInputs*nHiddens; ++n)
    w_[n] = std::complex<FloatType>(randw(ran), randw(ran));
  for (int i=0; i<nInputs; ++i)
    a_[i] = std::complex<FloatType>(randa(ran), randa(ran));
  for (int j=0; j<nHiddens; ++j)
    b_[j] = std::complex<FloatType>(randb(ran), randb(ran));
}

template <typename FloatType>
void ComplexRBM<FloatType>::update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate)
{
  for (int i=0; i<variables_.size(); ++i)
    variables_[i] = variables_[i] - learningRate*derivativeLoss[i];
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(knHiddens, knChains, kone, b_, &koneChains[0], &y_[0]);
  blas::gemm(knHiddens, knChains, knInputs, kone, kone, w_, &spinStates_[0], &y_[0]);
  // sa_k = \sum_i a_i*spinStates_ki
  blas::gemm(1, knChains, knInputs, kone, kzero, a_, &spinStates_[0], &sa_[0]);
}

template <typename FloatType>
void ComplexRBM<FloatType>::initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates)
{
  if (spinStates == NULL)
  {
    for (int i=0; i<spinStates_.size(); ++i)
      spinStates_[i] = 1.0;
  }
  else
  {
    for (int i=0; i<spinStates_.size(); ++i)
      spinStates_[i] = spinStates[i];
  }
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(knHiddens, knChains, kone, b_, &koneChains[0], &y_[0]);
  blas::gemm(knHiddens, knChains, knInputs, kone, kone, w_, &spinStates_[0], &y_[0]);
  // ly_kj = ln(cosh(y_kj))
  for (int j=0; j<y_.size(); ++j)
    ly_[j] = std::log(std::cosh(y_[j]));
  // sa_k = \sum_i a_i*spinStates_ki
  blas::gemm(1, knChains, knInputs, kone, kzero, a_, &spinStates_[0], &sa_[0]);
  // lnpsi_k = \sum_j ly_kj + sa_k
  for (int k=0; k<knChains; ++k)
    lnpsi[k] = sa_[k];
  blas::gemm(1, knChains, knHiddens, kone, kone, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void ComplexRBM<FloatType>::forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  index_ = spinFlipIndex;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<knHiddens; ++j)
      ly_[k*knHiddens+j] = std::log(std::cosh(y_[k*knHiddens+j]-w_[index_*knHiddens+j]*(ktwo*spinStates_[k*knInputs+index_].real())));
  // lnpsi_k = \sum_j ly_kj + \sum_i a_i*spinStates_ki
  for (int k=0; k<knChains; ++k)
    lnpsi[k] = sa_[k]-(ktwo*spinStates_[k*knInputs+index_].real())*a_[index_];
  blas::gemm(1, knChains, knHiddens, kone, kone, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void ComplexRBM<FloatType>::backward(std::complex<FloatType> * lnpsiGradients)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvsize = k*variables_.size(), kisize = k*knInputs, khsize = k*knHiddens;
    for (int i=0; i<knInputs; ++i)
      d_da_[kvsize+i] = spinStates_[kisize+i];
    for (int j=0; j<knHiddens; ++j)
      d_db_[kvsize+j] = std::tanh(y_[khsize+j]);
    for (int i=0; i<knInputs; ++i)
      for (int j=0; j<knHiddens; ++j)
        d_dw_[kvsize+i*knHiddens+j] = spinStates_[kisize+i].real()*d_db_[kvsize+j];
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*knChains);
}

template <typename FloatType>
void ComplexRBM<FloatType>::backward(std::complex<FloatType> * lnpsiGradients, const int & nChains)
{
  #pragma omp parallel for
  for (int k=0; k<nChains; ++k)
  {
    const int kvsize = k*variables_.size(), kisize = k*knInputs, khsize = k*knHiddens;
    for (int i=0; i<knInputs; ++i)
      d_da_[kvsize+i] = spinStates_[kisize+i];
    for (int j=0; j<knHiddens; ++j)
      d_db_[kvsize+j] = std::tanh(y_[khsize+j]);
    for (int i=0; i<knInputs; ++i)
      for (int j=0; j<knHiddens; ++j)
        d_dw_[kvsize+i*knHiddens+j] = spinStates_[kisize+i].real()*d_db_[kvsize+j];
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*nChains);
}

template <typename FloatType>
void ComplexRBM<FloatType>::load(const RBMDataType typeInfo, const std::string filePath)
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
  if (typeInfo == RBMDataType::W)
  {
    if (rawdata.size() == knInputs*knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        w_[i] = rawdata[i];
    else
      std::cout << " check 'w' size... " << std::endl;
  }
  else if (typeInfo == RBMDataType::V)
  {
    if (rawdata.size() == knInputs)
      for (int i=0; i<rawdata.size(); ++i)
        a_[i] = rawdata[i];
    else
      std::cout << " check 'a' size... " << std::endl;
  }
  else if (typeInfo == RBMDataType::H)
  {
    if (rawdata.size() == knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        b_[i] = rawdata[i];
    else
      std::cout << " check 'b' size... " << std::endl;
  }
}

template <typename FloatType>
void ComplexRBM<FloatType>::save(const RBMDataType typeInfo, const std::string filePath, const int precision) const
{
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  if (typeInfo == RBMDataType::W)
  {
    for (int i=0; i<knInputs; ++i)
    {
      for (int j=0; j<knHiddens; ++j)
        writer << w_[i*knHiddens+j] << " ";
      writer << std::endl;
    }
  }
  else if (typeInfo == RBMDataType::V)
  {
    for (int i=0; i<knInputs; ++i)
      writer << a_[i] << " ";
    writer << std::endl;
  }
  else if (typeInfo == RBMDataType::H)
    for (int j=0; j<knHiddens; ++j)
      writer << b_[j] << " ";
  writer.close();
}

template <typename FloatType>
void ComplexRBM<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    for (int j=0; j<knHiddens; ++j)
      y_[k*knHiddens+j] = y_[k*knHiddens+j]-w_[index_*knHiddens+j]*(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index_].real());
    sa_[k] = sa_[k]-(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index_].real())*a_[index_];
    spinStates_[k*knInputs+index_] = (kone.real()-ktwoTrueFalse[doSpinFlip[k]])*spinStates_[k*knInputs+index_].real();
  }
}

template <typename FloatType>
void ComplexRBM<FloatType>::swap_states(const int & k1, const int & k2)
{
  for (int i=0; i<knInputs; ++i)
    std::swap(spinStates_[k1*knInputs+i], spinStates_[k2*knInputs+i]);
  for (int j=0; j<knHiddens; ++j)
    std::swap(y_[k1*knHiddens+j], y_[k2*knHiddens+j]);
  std::swap(sa_[k1], sa_[k2]);
}


template <typename FloatType>
ComplexFNN<FloatType>::ComplexFNN(const int nInputs, const int nHiddens, const int nChains):
  knInputs(nInputs),
  knHiddens(nHiddens),
  knChains(nChains),
  kzero(std::complex<FloatType>(0.0, 0.0)),
  kone(std::complex<FloatType>(1.0, 0.0)),
  ktwo(2.0),
  koneChains(nChains, std::complex<FloatType>(1.0, 0.0)),
  ktwoTrueFalse({0.0, 2.0}),
  variables_(nInputs*nHiddens + 2*nHiddens),
  lnpsiGradients_(nChains*(nInputs*nHiddens + 2*nHiddens)),
  spinStates_(nInputs*nChains),
  y_(nHiddens*nChains),
  acty_(nHiddens*nChains),
  index_(0)
{
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double> randwi1(0, std::sqrt(1.0/(nInputs+nHiddens))),
                                   randb1(0, std::sqrt(1.0/nHiddens)),
                                   randw1o(0, std::sqrt(1.0/nHiddens));
  wi1_ = &variables_[0];
  b1_ = &variables_[nInputs*nHiddens];
  w1o_ = &variables_[nInputs*nHiddens + nHiddens];
  d_dwi1_ = &lnpsiGradients_[0];
  d_db1_ = &lnpsiGradients_[nInputs*nHiddens];
  d_dw1o_ = &lnpsiGradients_[nInputs*nHiddens + nHiddens];
  for (int i=0; i<nInputs*nHiddens; ++i)
    wi1_[i] = std::complex<FloatType>(randwi1(ran), randwi1(ran));
  for (int j=0; j<nHiddens; ++j)
    b1_[j] = std::complex<FloatType>(randb1(ran), randb1(ran));
  for (int j=0; j<nHiddens; ++j)
    w1o_[j] = std::complex<FloatType>(randw1o(ran), randw1o(ran));
}

template <typename FloatType>
void ComplexFNN<FloatType>::update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate)
{
  for (int i=0; i<variables_.size(); ++i)
    variables_[i] = variables_[i] - learningRate*derivativeLoss[i];
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(knHiddens, knChains, kone, b1_, &koneChains[0], &y_[0]);
  blas::gemm(knHiddens, knChains, knInputs, kone, kone, wi1_, &spinStates_[0], &y_[0]);
}

template <typename FloatType>
void ComplexFNN<FloatType>::initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates)
{
  if (spinStates == NULL)
  {
    for (int i=0; i<spinStates_.size(); ++i)
      spinStates_[i] = 1.0;
  }
  else
  {
    for (int i=0; i<spinStates_.size(); ++i)
      spinStates_[i] = spinStates[i];
  }
  // y_kj = \sum_i spinStates_ki wi1_ij + koneChains_k (x) b1_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(knHiddens, knChains, kone, b1_, &koneChains[0], &y_[0]);
  blas::gemm(knHiddens, knChains, knInputs, kone, kone, wi1_, &spinStates_[0], &y_[0]);
  // acty_kj = ln(cosh(y_kj))
  #pragma omp parallel for
  for (int j=0; j<y_.size(); ++j)
    acty_[j] = std::log(std::cosh(y_[j]));
  // lnpsi_k = \sum_j acty_kj w1o_j
  blas::gemm(1, knChains, knHiddens, kone, kzero, w1o_, &acty_[0], lnpsi);
}

template <typename FloatType>
void ComplexFNN<FloatType>::forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  index_ = spinFlipIndex;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<knHiddens; ++j)
      acty_[k*knHiddens+j] = std::log(std::cosh(y_[k*knHiddens+j]-wi1_[index_*knHiddens+j]*(ktwo*spinStates_[k*knInputs+index_].real())));
  // lnpsi_k = \sum_j acty_kj w1o_j
  blas::gemm(1, knChains, knHiddens, kone, kzero, w1o_, &acty_[0], lnpsi);
}

template <typename FloatType>
void ComplexFNN<FloatType>::backward(std::complex<FloatType> * lnpsiGradients)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvsize = k*variables_.size(), kisize = k*knInputs, khsize = k*knHiddens;
    for (int j=0; j<knHiddens; ++j)
      d_dw1o_[kvsize+j] = std::log(std::cosh(y_[khsize+j]));
    for (int j=0; j<knHiddens; ++j)
    {
      const std::complex<FloatType> tany_j = std::tanh(y_[khsize+j]);
      for (int i=0; i<knInputs; ++i)
        d_dwi1_[kvsize+i*knHiddens+j] = tany_j*spinStates_[kisize+i].real()*w1o_[j];
      d_db1_[kvsize+j] = tany_j*w1o_[j];
    }
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*knChains);
}

template <typename FloatType>
void ComplexFNN<FloatType>::backward(std::complex<FloatType> * lnpsiGradients, const int & nChains)
{
  #pragma omp parallel for
  for (int k=0; k<nChains; ++k)
  {
    const int kvsize = k*variables_.size(), kisize = k*knInputs, khsize = k*knHiddens;
    for (int j=0; j<knHiddens; ++j)
      d_dw1o_[kvsize+j] = std::log(std::cosh(y_[khsize+j]));
    for (int j=0; j<knHiddens; ++j)
    {
      const std::complex<FloatType> tany_j = std::tanh(y_[khsize+j]);
      for (int i=0; i<knInputs; ++i)
        d_dwi1_[kvsize+i*knHiddens+j] = tany_j*spinStates_[kisize+i].real()*w1o_[j];
      d_db1_[kvsize+j] = tany_j*w1o_[j];
    }
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*nChains);
}

template <typename FloatType>
void ComplexFNN<FloatType>::load(const FNNDataType typeInfo, const std::string filePath)
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
  if (typeInfo == FNNDataType::W1)
  {
    if (rawdata.size() == knInputs*knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        wi1_[i] = rawdata[i];
    else
      std::cout << " check 'w1' size... " << std::endl;
  }
  else if (typeInfo == FNNDataType::W2)
  {
    if (rawdata.size() == knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        w1o_[i] = rawdata[i];
    else
      std::cout << " check 'w2' size... " << std::endl;
  }
  else if (typeInfo == FNNDataType::B1)
  {
    if (rawdata.size() == knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        b1_[i] = rawdata[i];
    else
      std::cout << " check 'b1' size... " << std::endl;
  }
}

template <typename FloatType>
void ComplexFNN<FloatType>::save(const FNNDataType typeInfo, const std::string filePath,
  const int precision) const
{
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  if (typeInfo == FNNDataType::W1)
  {
    for (int i=0; i<knInputs; ++i)
    {
      for (int j=0; j<knHiddens; ++j)
        writer << wi1_[i*knHiddens+j] << " ";
      writer << std::endl;
    }
  }
  else if (typeInfo == FNNDataType::W2)
  {
    for (int j=0; j<knHiddens; ++j)
      writer << w1o_[j] << " ";
    writer << std::endl;
  }
  else if (typeInfo == FNNDataType::B1)
    for (int j=0; j<knHiddens; ++j)
      writer << b1_[j] << " ";
  writer.close();
}

template <typename FloatType>
void ComplexFNN<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    for (int j=0; j<knHiddens; ++j)
      y_[k*knHiddens+j] = y_[k*knHiddens+j]-wi1_[index_*knHiddens+j]*(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index_].real());
    spinStates_[k*knInputs+index_] = (kone.real()-ktwoTrueFalse[doSpinFlip[k]])*spinStates_[k*knInputs+index_].real();
  }
}

template <typename FloatType>
void ComplexFNN<FloatType>::swap_states(const int & k1, const int & k2)
{
  for (int i=0; i<knInputs; ++i)
    std::swap(spinStates_[k1*knInputs+i], spinStates_[k2*knInputs+i]);
  for (int j=0; j<knHiddens; ++j)
    std::swap(y_[k1*knHiddens+j], y_[k2*knHiddens+j]);
}
} // namespace spinhalf
