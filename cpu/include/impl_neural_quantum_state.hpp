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
RBM<FloatType>::RBM(const int nInputs, const int nHiddens, const int nChains):
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
  std::normal_distribution<double>
    randw(0, std::sqrt(1.0/(nInputs+nHiddens))),
    randa(0, std::sqrt(1.0/nInputs)),
    randb(0, std::sqrt(1.0/nHiddens));
  w_ = &variables_[0],
  a_ = &variables_[knInputs*knHiddens],
  b_ = &variables_[knInputs*knHiddens+nInputs],
  d_dw_ = &lnpsiGradients_[0],
  d_da_ = &lnpsiGradients_[knInputs*knHiddens],
  d_db_ = &lnpsiGradients_[knInputs*knHiddens+nInputs];
  for (int n=0; n<nInputs*nHiddens; ++n)
    w_[n] = std::complex<FloatType>(1e-1*randw(ran), 1e-1*randw(ran));
  for (int i=0; i<nInputs; ++i)
    a_[i] = kzero;
  for (int j=0; j<nHiddens; ++j)
    b_[j] = std::complex<FloatType>(1e-1*randb(ran), 1e-1*randb(ran));
}

template <typename FloatType>
void RBM<FloatType>::update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate)
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
void RBM<FloatType>::update_partial_variables(const std::complex<FloatType> * derivativeLoss,
  const FloatType learningRate, const std::vector<int> & hiddenNodes)
{
  // * index notation
  // hiddenNodes = [j_0, j_1, j_2, ...]
  // derivativeLoss = [a_0, a_1, a_2,... b_j_1, b_j_2, b_j_3,..., w_0j_0, w_0j_1, w_0j_2,..., w_1j_0, w_1j_1, w_1j_2,...]
  int idx = 0;
  for (int i=0; i<knInputs; ++i)
    a_[i] = a_[i]-learningRate*derivativeLoss[idx++];
  for (const auto & j : hiddenNodes)
    b_[j] = b_[j]-learningRate*derivativeLoss[idx++];
  for (int i=0; i<knInputs; ++i)
    for (const auto & j : hiddenNodes)
      w_[i*knHiddens+j] = w_[i*knHiddens+j]-learningRate*derivativeLoss[idx++];
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(knHiddens, knChains, kone, b_, &koneChains[0], &y_[0]);
  blas::gemm(knHiddens, knChains, knInputs, kone, kone, w_, &spinStates_[0], &y_[0]);
  // sa_k = \sum_i a_i*spinStates_ki
  blas::gemm(1, knChains, knInputs, kone, kzero, a_, &spinStates_[0], &sa_[0]);
}

template <typename FloatType>
void RBM<FloatType>::initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates)
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
    ly_[j] = logcosh(y_[j]);
  // sa_k = \sum_i a_i*spinStates_ki
  blas::gemm(1, knChains, knInputs, kone, kzero, a_, &spinStates_[0], &sa_[0]);
  // lnpsi_k = \sum_j ly_kj + sa_k
  for (int k=0; k<knChains; ++k)
    lnpsi[k] = sa_[k];
  blas::gemm(1, knChains, knHiddens, kone, kone, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void RBM<FloatType>::forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  index_ = spinFlipIndex;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<knHiddens; ++j)
      ly_[k*knHiddens+j] = logcosh(y_[k*knHiddens+j]-w_[index_*knHiddens+j]*(ktwo*spinStates_[k*knInputs+index_].real()));
  // lnpsi_k = \sum_j ly_kj + \sum_i a_i*spinStates_ki
  for (int k=0; k<knChains; ++k)
    lnpsi[k] = sa_[k]-(ktwo*spinStates_[k*knInputs+index_].real())*a_[index_];
  blas::gemm(1, knChains, knHiddens, kone, kone, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void RBM<FloatType>::forward(const std::vector<int> & spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  ly_ = y_;
  for (const auto & index : spinFlipIndex)
    for (int k=0; k<knChains; ++k)
      for (int j=0; j<knHiddens; ++j)
        ly_[k*knHiddens+j] -= w_[index*knHiddens+j]*(ktwo*spinStates_[k*knInputs+index].real());
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<knHiddens; ++j)
      ly_[k*knHiddens+j] = logcosh(ly_[k*knHiddens+j]);
  std::memcpy(lnpsi, &sa_[0], sizeof(std::complex<FloatType>)*knChains);
  // lnpsi_k = \sum_j ly_kj + \sum_i a_i*spinStates_ki
  for (const auto & index : spinFlipIndex)
    for (int k=0; k<knChains; ++k)
      lnpsi[k] -= (ktwo*spinStates_[k*knInputs+index].real())*a_[index];
  blas::gemm(1, knChains, knHiddens, kone, kone, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void RBM<FloatType>::forward(const std::vector<std::vector<int> > & spinFlipIndexes, std::complex<FloatType> * lnpsi)
{
  ly_ = y_;
  for (int k=0; k<knChains; ++k)
    for (const auto & index : spinFlipIndexes[k])
      for (int j=0; j<knHiddens; ++j)
        ly_[k*knHiddens+j] -= w_[index*knHiddens+j]*(ktwo*spinStates_[k*knInputs+index].real());
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<knHiddens; ++j)
      ly_[k*knHiddens+j] = logcosh(ly_[k*knHiddens+j]);
  std::memcpy(lnpsi, &sa_[0], sizeof(std::complex<FloatType>)*knChains);
  // lnpsi_k = \sum_j ly_kj + \sum_i a_i*spinStates_ki
  for (int k=0; k<knChains; ++k)
    for (const auto & index : spinFlipIndexes[k])
      lnpsi[k] -= (ktwo*spinStates_[k*knInputs+index].real())*a_[index];
  blas::gemm(1, knChains, knHiddens, kone, kone, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void RBM<FloatType>::backward(std::complex<FloatType> * lnpsiGradients)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*knHiddens;
    for (int i=0; i<knInputs; ++i)
      d_da_[kvSize+i] = spinStates_[kiSize+i];
    for (int j=0; j<knHiddens; ++j)
      d_db_[kvSize+j] = std::tanh(y_[khSize+j]);
    for (int i=0; i<knInputs; ++i)
      for (int j=0; j<knHiddens; ++j)
        d_dw_[kvSize+i*knHiddens+j] = spinStates_[kiSize+i].real()*d_db_[kvSize+j];
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*knChains);
}

template <typename FloatType>
void RBM<FloatType>::partial_backward(std::complex<FloatType> * lnpsiGradients, const int & nChains)
{
  #pragma omp parallel for
  for (int k=0; k<nChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*knHiddens;
    for (int i=0; i<knInputs; ++i)
      d_da_[kvSize+i] = spinStates_[kiSize+i];
    for (int j=0; j<knHiddens; ++j)
      d_db_[kvSize+j] = std::tanh(y_[khSize+j]);
    for (int i=0; i<knInputs; ++i)
      for (int j=0; j<knHiddens; ++j)
        d_dw_[kvSize+i*knHiddens+j] = spinStates_[kiSize+i].real()*d_db_[kvSize+j];
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*nChains);
}

template <typename FloatType>
void RBM<FloatType>::partial_backward(std::complex<FloatType> * lnpsiGradients, const std::vector<int> & hiddenNodes)
{
  /* index notation
     hiddenNodes = [j_0, j_1, j_2, ...]
     lnpsiGradients_k = [d_da_k0, d_da_k1, d_da_k2,... d_db_kj_0, d_db_kj_1, d_db_kj_2,...,
                         d_dw_k0j_0, d_dw_k0j_1, d_dw_k0j_2,..., d_dw_k1j_0, d_dw_k1j_2,...] */
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*knHiddens;
    for (int i=0; i<knInputs; ++i)
      d_da_[kvSize+i] = spinStates_[kiSize+i];
    for (const auto & j : hiddenNodes)
      d_db_[kvSize+j] = std::tanh(y_[khSize+j]);
    for (int i=0; i<knInputs; ++i)
      for (const auto & j : hiddenNodes)
        d_dw_[kvSize+i*knHiddens+j] = spinStates_[kiSize+i].real()*d_db_[kvSize+j];
  }
  // save the results into lnpsiGradients
  int idx = 0;
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*knHiddens;
    for (int i=0; i<knInputs; ++i)
      lnpsiGradients[idx++] = d_da_[kvSize+i];
    for (const auto & j : hiddenNodes)
      lnpsiGradients[idx++] = d_db_[kvSize+j];
    for (int i=0; i<knInputs; ++i)
      for (const auto & j : hiddenNodes)
        lnpsiGradients[idx++] = d_dw_[kvSize+i*knHiddens+j];
  }
}

template <typename FloatType>
void RBM<FloatType>::load(const RBMDataType typeInfo, const std::string filePath)
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
void RBM<FloatType>::save(const RBMDataType typeInfo, const std::string filePath, const int precision) const
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
void RBM<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
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
void RBM<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip, const std::vector<std::vector<int> > & spinFlipIndexes)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    for (int j=0; j<knHiddens; ++j)
      for (const auto & index : spinFlipIndexes[k])
        y_[k*knHiddens+j] = y_[k*knHiddens+j]-w_[index*knHiddens+j]*(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index].real());
    for (const auto & index : spinFlipIndexes[k])
    {
      sa_[k] = sa_[k]-(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index].real())*a_[index];
      spinStates_[k*knInputs+index] = (kone.real()-ktwoTrueFalse[doSpinFlip[k]])*spinStates_[k*knInputs+index].real();
    }
  }
}

template <typename FloatType>
void RBM<FloatType>::swap_states(const int & k1, const int & k2)
{
  for (int i=0; i<knInputs; ++i)
    std::swap(spinStates_[k1*knInputs+i], spinStates_[k2*knInputs+i]);
  for (int j=0; j<knHiddens; ++j)
    std::swap(y_[k1*knHiddens+j], y_[k2*knHiddens+j]);
  std::swap(sa_[k1], sa_[k2]);
}


template <typename FloatType>
RBMTrSymm<FloatType>::RBMTrSymm(const int nInputs, const int alpha, const int nChains):
  knInputs(nInputs),
  kAlpha(alpha),
  knChains(nChains),
  variables_(nInputs*alpha+1+alpha),
  lnpsiGradients_(nChains*(nInputs*alpha+1+alpha)),
  spinStates_(nInputs*nChains),
  y_(nInputs*alpha*nChains),
  ly_(nInputs*alpha*nChains),
  sa_(nChains),
  wf_(nInputs*nInputs*alpha),
  bf_(nInputs*alpha),
  af_(nInputs),
  kzero(std::complex<FloatType>(0.0, 0.0)),
  kone(std::complex<FloatType>(1.0, 0.0)),
  ktwo(2.0),
  koneChains(nChains, std::complex<FloatType>(1.0,0.0)),
  koneHiddens(nInputs*alpha, std::complex<FloatType>(1.0,0.0)),
  ktwoTrueFalse({0.0, 2.0}),
  index_(0)
{
  if (alpha <= 0)
    throw std::invalid_argument("alpha <= 0 --> alpha should be equal or larger than 1.");
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double>
    randw(0, std::sqrt(1.0/((1+kAlpha)*nInputs))),
    randb(0, std::sqrt(1.0/(knInputs*kAlpha)));
  w_ = &variables_[0],
  a_ = &variables_[knInputs*kAlpha],
  b_ = &variables_[knInputs*kAlpha+1],
  d_dw_ = &lnpsiGradients_[0],
  d_da_ = &lnpsiGradients_[knInputs*kAlpha],
  d_db_ = &lnpsiGradients_[knInputs*kAlpha+1];
  for (int n=0; n<knInputs*kAlpha; ++n)
    w_[n] = std::complex<FloatType>(1e-1*randw(ran), 1e-1*randw(ran));
  a_[0] = kzero;
  for (int f=0; f<kAlpha; ++f)
    b_[f] = std::complex<FloatType>(1e-1*randb(ran), 1e-1*randb(ran));
}

template <typename FloatType>
void RBMTrSymm<FloatType>::construct_weight_and_bias_()
{
  // i=0,1,...,N-1; j=0,1,...,N-1; f=0,...,alpha-1
  // wf_{i,f*nInputs+j} (=wf_{j,f*nInputs+i}), bf_[f*nInputs+j], af_[j]
  for (int f=0; f<kAlpha; ++f)
  {
    for (int j=0; j<knInputs; ++j)
    {
      for (int i=j; i<knInputs; ++i)
        wf_[i*knInputs*kAlpha+f*knInputs+j] = w_[f*knInputs+(i+j)%knInputs];
      for (int i=(j+1); i<knInputs; ++i)
        wf_[j*knInputs*kAlpha+f*knInputs+i] = wf_[i*knInputs*kAlpha+f*knInputs+j];
      bf_[f*knInputs+j] = b_[f];
    }
  }
  std::fill(af_.begin(), af_.end(), a_[0]);
}

template <typename FloatType>
void RBMTrSymm<FloatType>::update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate)
{
  for (int i=0; i<variables_.size(); ++i)
    variables_[i] = variables_[i] - learningRate*derivativeLoss[i];
  this->construct_weight_and_bias_();
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(kAlpha*knInputs, knChains, kone, &bf_[0], &koneChains[0], &y_[0]);
  blas::gemm(kAlpha*knInputs, knChains, knInputs, kone, kone, &wf_[0], &spinStates_[0], &y_[0]);
  // sa_k = \sum_i a_i*spinStates_ki
  blas::gemm(1, knChains, knInputs, kone, kzero, &af_[0], &spinStates_[0], &sa_[0]);
}

template <typename FloatType>
void RBMTrSymm<FloatType>::initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates)
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
  this->construct_weight_and_bias_();
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(kAlpha*knInputs, knChains, kone, &bf_[0], &koneChains[0], &y_[0]);
  blas::gemm(kAlpha*knInputs, knChains, knInputs, kone, kone, &wf_[0], &spinStates_[0], &y_[0]);
  // ly_kj = ln(cosh(y_kj))
  for (int j=0; j<y_.size(); ++j)
    ly_[j] = logcosh(y_[j]);
  // sa_k = \sum_i af_i*spinStates_ki
  blas::gemm(1, knChains, knInputs, kone, kzero, &af_[0], &spinStates_[0], &sa_[0]);
  // lnpsi_k = \sum_j ly_kj + sa_k
  for (int k=0; k<knChains; ++k)
    lnpsi[k] = sa_[k];
  blas::gemm(1, knChains, kAlpha*knInputs, kone, kone, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void RBMTrSymm<FloatType>::forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  index_ = spinFlipIndex;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<kAlpha*knInputs; ++j)
      ly_[k*kAlpha*knInputs+j] = logcosh(y_[k*kAlpha*knInputs+j]-wf_[index_*kAlpha*knInputs+j]*(ktwo*spinStates_[k*knInputs+index_].real()));
  // lnpsi_k = \sum_j ly_kj + \sum_i a_i*spinStates_ki
  for (int k=0; k<knChains; ++k)
    lnpsi[k] = sa_[k]-(ktwo*spinStates_[k*knInputs+index_].real())*a_[0];
  blas::gemm(1, knChains, kAlpha*knInputs, kone, kone, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void RBMTrSymm<FloatType>::backward(std::complex<FloatType> * lnpsiGradients)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs;
    d_da_[kvSize] = kzero;
    for (int i=0; i<knInputs; ++i)
      d_da_[kvSize] += spinStates_[kiSize+i].real();
    for (int f=0; f<kAlpha; ++f)
    {
      d_db_[kvSize+f] = kzero;
      for (int j=0; j<knInputs; ++j)
        d_db_[kvSize+f] += std::tanh(y_[k*kAlpha*knInputs+f*knInputs+j]);
      for (int i=0; i<knInputs; ++i)
      {
        d_dw_[kvSize+f*knInputs+i] = kzero;
        for (int j=0; j<knInputs; ++j)
          d_dw_[kvSize+f*knInputs+i] += std::tanh(y_[k*kAlpha*knInputs+f*knInputs+j])*spinStates_[kiSize+(knInputs+i-j)%knInputs].real();
      }
    }
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*lnpsiGradients_.size());
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
  if (rawdata.size() == variables_.size())
    variables_ = rawdata;
  else
    std::cout << " check parameter size... " << std::endl;
}

template <typename FloatType>
void RBMTrSymm<FloatType>::save(const std::string filePath, const int precision) const
{
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  for (const auto & var : variables_)
    writer << var << " ";
  writer.close();
}

template <typename FloatType>
void RBMTrSymm<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    for (int j=0; j<kAlpha*knInputs; ++j)
      y_[k*kAlpha*knInputs+j] = y_[k*kAlpha*knInputs+j]-wf_[index_*kAlpha*knInputs+j]*(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index_].real());
    sa_[k] = sa_[k]-(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index_].real())*a_[0];
    spinStates_[k*knInputs+index_] = (kone.real()-ktwoTrueFalse[doSpinFlip[k]])*spinStates_[k*knInputs+index_].real();
  }
}


template <typename FloatType>
RBMSfSymm<FloatType>::RBMSfSymm(const int nInputs, const int alpha, const int nChains):
  knInputs(nInputs),
  kAlpha(alpha),
  knChains(nChains),
  variables_(nInputs*alpha*nInputs),
  lnpsiGradients_(nChains*(nInputs*alpha*nInputs)),
  spinStates_(nInputs*nChains),
  y_(alpha*nInputs*nChains),
  ly_(alpha*nInputs*nChains),
  kzero(std::complex<FloatType>(0.0, 0.0)),
  kone(std::complex<FloatType>(1.0, 0.0)),
  ktwo(2.0),
  koneChains(nChains, std::complex<FloatType>(1.0,0.0)),
  koneHiddens(alpha*nInputs, std::complex<FloatType>(1.0,0.0)),
  ktwoTrueFalse({0.0, 2.0}),
  index_(0)
{
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double> randw(0, std::sqrt(1.0/((1+kAlpha)*nInputs)));
  w_ = &variables_[0];
  d_dw_ = &lnpsiGradients_[0];
  for (int n=0; n<knInputs*kAlpha*knInputs; ++n)
    w_[n] = std::complex<FloatType>(1e-1*randw(ran), 1e-1*randw(ran));
}

template <typename FloatType>
void RBMSfSymm<FloatType>::update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate)
{
  for (int i=0; i<variables_.size(); ++i)
    variables_[i] = variables_[i] - learningRate*derivativeLoss[i];
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  blas::gemm(kAlpha*knInputs, knChains, knInputs, kone, kzero, w_, &spinStates_[0], &y_[0]);
}

template <typename FloatType>
void RBMSfSymm<FloatType>::initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates)
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
  blas::gemm(kAlpha*knInputs, knChains, knInputs, kone, kzero, w_, &spinStates_[0], &y_[0]);
  // ly_kj = ln(cosh(y_kj))
  for (int j=0; j<y_.size(); ++j)
    ly_[j] = logcosh(y_[j]);
  blas::gemm(1, knChains, kAlpha*knInputs, kone, kzero, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void RBMSfSymm<FloatType>::forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  index_ = spinFlipIndex;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<kAlpha*knInputs; ++j)
      ly_[k*kAlpha*knInputs+j] = logcosh(y_[k*kAlpha*knInputs+j]-w_[index_*kAlpha*knInputs+j]*(ktwo*spinStates_[k*knInputs+index_].real()));
  // lnpsi_k = \sum_j ly_kj
  blas::gemm(1, knChains, kAlpha*knInputs, kone, kzero, &koneHiddens[0], &ly_[0], lnpsi);
}

template <typename FloatType>
void RBMSfSymm<FloatType>::backward(std::complex<FloatType> * lnpsiGradients)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*kAlpha*knInputs;
    for (int i=0; i<knInputs; ++i)
      for (int j=0; j<kAlpha*knInputs; ++j)
        d_dw_[kvSize+i*kAlpha*knInputs+j] = spinStates_[kiSize+i].real()*std::tanh(y_[khSize+j]);
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*knChains);
}

template <typename FloatType>
void RBMSfSymm<FloatType>::load(const std::string filePath)
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
  if (rawdata.size() == variables_.size())
    variables_ = rawdata;
  else
    std::cout << " check parameter size... " << std::endl;
}

template <typename FloatType>
void RBMSfSymm<FloatType>::save(const std::string filePath, const int precision) const
{
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  for (const auto & var : variables_)
    writer << var << " ";
  writer.close();
}

template <typename FloatType>
void RBMSfSymm<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    for (int j=0; j<kAlpha*knInputs; ++j)
      y_[k*kAlpha*knInputs+j] = y_[k*kAlpha*knInputs+j]-w_[index_*kAlpha*knInputs+j]*(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index_].real());
    spinStates_[k*knInputs+index_] = (kone.real()-ktwoTrueFalse[doSpinFlip[k]])*spinStates_[k*knInputs+index_].real();
  }
}



template <typename FloatType>
FFNN<FloatType>::FFNN(const int nInputs, const int nHiddens, const int nChains):
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
  std::normal_distribution<double>
    randwi1(0, std::sqrt(1.0/(nInputs+nHiddens))),
    randb1(0, std::sqrt(1.0/nHiddens)),
    randw1o(0, std::sqrt(1.0/nHiddens));
  wi1_ = &variables_[0];
  b1_ = &variables_[nInputs*nHiddens];
  w1o_ = &variables_[nInputs*nHiddens + nHiddens];
  d_dwi1_ = &lnpsiGradients_[0];
  d_db1_ = &lnpsiGradients_[nInputs*nHiddens];
  d_dw1o_ = &lnpsiGradients_[nInputs*nHiddens + nHiddens];
  for (int i=0; i<nInputs*nHiddens; ++i)
    wi1_[i] = std::complex<FloatType>(randwi1(ran), 1e-1*randwi1(ran));
  for (int j=0; j<nHiddens; ++j)
    b1_[j] = std::complex<FloatType>(randb1(ran), 1e-1*randb1(ran));
  for (int j=0; j<nHiddens; ++j)
    w1o_[j] = std::complex<FloatType>(randw1o(ran), 1e-1*randw1o(ran));
}

template <typename FloatType>
void FFNN<FloatType>::update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate)
{
  for (int i=0; i<variables_.size(); ++i)
    variables_[i] = variables_[i] - learningRate*derivativeLoss[i];
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(knHiddens, knChains, kone, b1_, &koneChains[0], &y_[0]);
  blas::gemm(knHiddens, knChains, knInputs, kone, kone, wi1_, &spinStates_[0], &y_[0]);
}

template <typename FloatType>
void FFNN<FloatType>::update_partial_variables(const std::complex<FloatType> * derivativeLoss,
  const FloatType learningRate, const std::vector<int> & hiddenNodes)
{
  // * index notation
  // hiddenNodes = [j_0, j_1, j_2, ...]
  // derivativeLoss = [wi1_0j_0, wi1_1j_0, wi1_2j_0,... wi1_0j_1, wi1_1j_1, wi1_2j_1,..., b1_j_0, b1_j_1, b1_j_2,..., w1o_j_0, w1o_j_1, w1o_j_2]
  int idx = 0;
  for (const int & j : hiddenNodes)
    for (int i=0; i<knInputs; ++i)
      wi1_[i*knHiddens+j] = wi1_[i*knHiddens+j]-learningRate*derivativeLoss[idx++];
  for (const int & j : hiddenNodes)
    b1_[j] = b1_[j]-learningRate*derivativeLoss[idx++];
  for (const int & j : hiddenNodes)
    w1o_[j] = w1o_[j]-learningRate*derivativeLoss[idx++];
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(knHiddens, knChains, kone, b1_, &koneChains[0], &y_[0]);
  blas::gemm(knHiddens, knChains, knInputs, kone, kone, wi1_, &spinStates_[0], &y_[0]);
}

template <typename FloatType>
void FFNN<FloatType>::initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates)
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
    acty_[j] = logcosh(y_[j]);
  // lnpsi_k = \sum_j acty_kj w1o_j
  blas::gemm(1, knChains, knHiddens, kone, kzero, w1o_, &acty_[0], lnpsi);
}

template <typename FloatType>
void FFNN<FloatType>::forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  index_ = spinFlipIndex;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<knHiddens; ++j)
      acty_[k*knHiddens+j] = logcosh(y_[k*knHiddens+j]-wi1_[index_*knHiddens+j]*(ktwo*spinStates_[k*knInputs+index_].real()));
  // lnpsi_k = \sum_j acty_kj w1o_j
  blas::gemm(1, knChains, knHiddens, kone, kzero, w1o_, &acty_[0], lnpsi);
}

template <typename FloatType>
void FFNN<FloatType>::forward(const std::vector<int> & spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  acty_ = y_;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<knHiddens; ++j)
    {
      for (const auto & index : spinFlipIndex)
        acty_[k*knHiddens+j] = acty_[k*knHiddens+j]-wi1_[index*knHiddens+j]*(ktwo*spinStates_[k*knInputs+index].real());
      acty_[k*knHiddens+j] = logcosh(acty_[k*knHiddens+j]);
    }
  // lnpsi_k = \sum_j acty_kj w1o_j
  blas::gemm(1, knChains, knHiddens, kone, kzero, w1o_, &acty_[0], lnpsi);
}

template <typename FloatType>
void FFNN<FloatType>::forward(const std::vector<std::vector<int> > & spinFlipIndexes, std::complex<FloatType> * lnpsi)
{
  acty_ = y_;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<knHiddens; ++j)
    {
      for (const auto & index : spinFlipIndexes[k])
        acty_[k*knHiddens+j] = acty_[k*knHiddens+j]-wi1_[index*knHiddens+j]*(ktwo*spinStates_[k*knInputs+index].real());
      acty_[k*knHiddens+j] = logcosh(acty_[k*knHiddens+j]);
    }
  // lnpsi_k = \sum_j acty_kj w1o_j
  blas::gemm(1, knChains, knHiddens, kone, kzero, w1o_, &acty_[0], lnpsi);
}

template <typename FloatType>
void FFNN<FloatType>::backward(std::complex<FloatType> * lnpsiGradients)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*knHiddens;
    for (int j=0; j<knHiddens; ++j)
      d_dw1o_[kvSize+j] = logcosh(y_[khSize+j]);
    for (int j=0; j<knHiddens; ++j)
    {
      const std::complex<FloatType> tany_j = std::tanh(y_[khSize+j]);
      for (int i=0; i<knInputs; ++i)
        d_dwi1_[kvSize+i*knHiddens+j] = tany_j*spinStates_[kiSize+i].real()*w1o_[j];
      d_db1_[kvSize+j] = tany_j*w1o_[j];
    }
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*knChains);
}

template <typename FloatType>
void FFNN<FloatType>::partial_backward(std::complex<FloatType> * lnpsiGradients, const int & nChains)
{
  #pragma omp parallel for
  for (int k=0; k<nChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*knHiddens;
    for (int j=0; j<knHiddens; ++j)
      d_dw1o_[kvSize+j] = logcosh(y_[khSize+j]);
    for (int j=0; j<knHiddens; ++j)
    {
      const std::complex<FloatType> tany_j = std::tanh(y_[khSize+j]);
      for (int i=0; i<knInputs; ++i)
        d_dwi1_[kvSize+i*knHiddens+j] = tany_j*spinStates_[kiSize+i].real()*w1o_[j];
      d_db1_[kvSize+j] = tany_j*w1o_[j];
    }
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*nChains);
}

template <typename FloatType>
void FFNN<FloatType>::partial_backward(std::complex<FloatType> * lnpsiGradients, const std::vector<int> & hiddenNodes)
{
  /* index notation
     hiddenNodes = [j_0, j_1, j_2, ...]
     lnpsiGradients_k = [d_dwi1_k0j_0, d_dwi1_k1j_0, d_dwi1_k2j_0,... d_dwi1_k0j_1, d_dwi1_k1j_1, d_dwi1_k2j_1,...,
                         d_db1_kj_0, d_db1_kj_1, d_db1_kj_2,..., d_dw1o_kj_0, d_dw1o_kj_1, d_dw1o_kj_2,...] */
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*knHiddens;
    for (const int & j : hiddenNodes)
    {
      const std::complex<FloatType> & tany_j = std::tanh(y_[khSize+j]);
      for (int i=0; i<knInputs; ++i)
        d_dwi1_[kvSize+i*knHiddens+j] = tany_j*spinStates_[kiSize+i].real()*w1o_[j];
      d_db1_[kvSize+j] = tany_j*w1o_[j];
      d_dw1o_[kvSize+j] = logcosh(y_[khSize+j]);
    }
  }
  // save the results into lnpsiGradients
  int idx = 0;
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*knHiddens;
    for (const int & j : hiddenNodes)
      for (int i=0; i<knInputs; ++i)
        lnpsiGradients[idx++] = d_dwi1_[kvSize+i*knHiddens+j];
    for (const int & j : hiddenNodes)
      lnpsiGradients[idx++] = d_db1_[kvSize+j];
    for (const int & j : hiddenNodes)
      lnpsiGradients[idx++] = d_dw1o_[kvSize+j];
  }
}

template <typename FloatType>
void FFNN<FloatType>::load(const FFNNDataType typeInfo, const std::string filePath)
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
  if (typeInfo == FFNNDataType::W1)
  {
    if (rawdata.size() == knInputs*knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        wi1_[i] = rawdata[i];
    else
      std::cout << " check 'w1' size... " << std::endl;
  }
  else if (typeInfo == FFNNDataType::W2)
  {
    if (rawdata.size() == knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        w1o_[i] = rawdata[i];
    else
      std::cout << " check 'w2' size... " << std::endl;
  }
  else if (typeInfo == FFNNDataType::B1)
  {
    if (rawdata.size() == knHiddens)
      for (int i=0; i<rawdata.size(); ++i)
        b1_[i] = rawdata[i];
    else
      std::cout << " check 'b1' size... " << std::endl;
  }
}

template <typename FloatType>
void FFNN<FloatType>::save(const FFNNDataType typeInfo, const std::string filePath, const int precision) const
{
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  if (typeInfo == FFNNDataType::W1)
  {
    for (int i=0; i<knInputs; ++i)
    {
      for (int j=0; j<knHiddens; ++j)
        writer << wi1_[i*knHiddens+j] << " ";
      writer << std::endl;
    }
  }
  else if (typeInfo == FFNNDataType::W2)
  {
    for (int j=0; j<knHiddens; ++j)
      writer << w1o_[j] << " ";
    writer << std::endl;
  }
  else if (typeInfo == FFNNDataType::B1)
    for (int j=0; j<knHiddens; ++j)
      writer << b1_[j] << " ";
  writer.close();
}

template <typename FloatType>
void FFNN<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    for (int j=0; j<knHiddens; ++j)
      y_[k*knHiddens+j] = y_[k*knHiddens+j]-wi1_[index_*knHiddens+j]*(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index_].real());
    spinStates_[k*knInputs+index_] = (kone.real()-ktwoTrueFalse[doSpinFlip[k]])*spinStates_[k*knInputs+index_].real();
  }
}

template <typename FloatType>
void FFNN<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip,
  const std::vector<std::vector<int> > & spinFlipIndexes)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    for (int j=0; j<knHiddens; ++j)
      for (const auto & index : spinFlipIndexes[k])
        y_[k*knHiddens+j] = y_[k*knHiddens+j]-wi1_[index*knHiddens+j]*(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index].real());
    for (const auto & index : spinFlipIndexes[k])
      spinStates_[k*knInputs+index] = (kone.real()-ktwoTrueFalse[doSpinFlip[k]])*spinStates_[k*knInputs+index].real();
  }
}

template <typename FloatType>
void FFNN<FloatType>::swap_states(const int & k1, const int & k2)
{
  for (int i=0; i<knInputs; ++i)
    std::swap(spinStates_[k1*knInputs+i], spinStates_[k2*knInputs+i]);
  for (int j=0; j<knHiddens; ++j)
    std::swap(y_[k1*knHiddens+j], y_[k2*knHiddens+j]);
}


template <typename FloatType>
FFNNTrSymm<FloatType>::FFNNTrSymm(const int nInputs, const int alpha, const int nChains):
  knInputs(nInputs),
  kAlpha(alpha),
  knChains(nChains),
  kzero(std::complex<FloatType>(0.0, 0.0)),
  kone(std::complex<FloatType>(1.0, 0.0)),
  ktwo(2.0),
  koneChains(nChains, std::complex<FloatType>(1.0, 0.0)),
  ktwoTrueFalse({0.0, 2.0}),
  variables_(nInputs*alpha+2*alpha),
  lnpsiGradients_(nChains*(nInputs*alpha+2*alpha)),
  spinStates_(nInputs*nChains),
  y_(alpha*nInputs*nChains),
  acty_(alpha*nInputs*nChains),
  wf1_(nInputs*nInputs*alpha),
  wf2_(nInputs*alpha),
  bf_(nInputs*alpha),
  index_(0)
{
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double>
    randwi1(0, std::sqrt(1.0/((1+kAlpha)*knInputs))),
    randb1(0, std::sqrt(1.0/(kAlpha*knInputs))),
    randw1o(0, std::sqrt(1.0/(kAlpha*knInputs)));
  wi1_ = &variables_[0];
  b1_ = &variables_[nInputs*kAlpha];
  w1o_ = &variables_[nInputs*kAlpha + kAlpha];
  d_dwi1_ = &lnpsiGradients_[0];
  d_db1_ = &lnpsiGradients_[nInputs*kAlpha];
  d_dw1o_ = &lnpsiGradients_[nInputs*kAlpha + kAlpha];
  for (int i=0; i<nInputs*kAlpha; ++i)
    wi1_[i] = std::complex<FloatType>(randwi1(ran), 1e-1*randwi1(ran));
  for (int j=0; j<kAlpha; ++j)
    b1_[j] = std::complex<FloatType>(randb1(ran), 1e-1*randb1(ran));
  for (int j=0; j<kAlpha; ++j)
    w1o_[j] = std::complex<FloatType>(randw1o(ran), 1e-1*randw1o(ran));
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::construct_weight_and_bias_()
{
  // i=0,1,...,N-1; j=0,1,...,N-1; f=0,...,alpha-1
  // wf1_{i,f*nInputs+j} (=wf1_{j,f*nInputs+i}), bf_[f*nInputs+j], wf2_{f*nInputs+j}
  for (int f=0; f<kAlpha; ++f)
  {
    for (int j=0; j<knInputs; ++j)
    {
      for (int i=j; i<knInputs; ++i)
        wf1_[i*knInputs*kAlpha+f*knInputs+j] = wi1_[f*knInputs+(i+j)%knInputs];
      for (int i=(j+1); i<knInputs; ++i)
        wf1_[j*knInputs*kAlpha+f*knInputs+i] = wf1_[i*knInputs*kAlpha+f*knInputs+j];
      bf_[f*knInputs+j] = b1_[f];
      wf2_[f*knInputs+j] = w1o_[f];
    }
  }
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate)
{
  for (int i=0; i<variables_.size(); ++i)
    variables_[i] = variables_[i] - learningRate*derivativeLoss[i];
  this->construct_weight_and_bias_();
  // y_kj = \sum_i spinStates_ki wf1_ij + koneChains_k (x) b_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(kAlpha*knInputs, knChains, kone, &bf_[0], &koneChains[0], &y_[0]);
  blas::gemm(kAlpha*knInputs, knChains, knInputs, kone, kone, &wf1_[0], &spinStates_[0], &y_[0]);
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates)
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
  this->construct_weight_and_bias_();
  // y_kj = \sum_i spinStates_ki wf1_ij + koneChains_k (x) bf_j
  std::fill(y_.begin(), y_.end(), kzero);
  blas::ger(kAlpha*knInputs, knChains, kone, &bf_[0], &koneChains[0], &y_[0]);
  blas::gemm(kAlpha*knInputs, knChains, knInputs, kone, kone, &wf1_[0], &spinStates_[0], &y_[0]);
  // acty_kj = ln(cosh(y_kj))
  #pragma omp parallel for
  for (int j=0; j<y_.size(); ++j)
    acty_[j] = logcosh(y_[j]);
  // lnpsi_k = \sum_j acty_kj wf2_j
  blas::gemm(1, knChains, kAlpha*knInputs, kone, kzero, &wf2_[0], &acty_[0], lnpsi);
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  index_ = spinFlipIndex;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<kAlpha*knInputs; ++j)
      acty_[k*kAlpha*knInputs+j] = logcosh(y_[k*kAlpha*knInputs+j]-wf1_[index_*kAlpha*knInputs+j]*(ktwo*spinStates_[k*knInputs+index_].real()));
  // lnpsi_k = \sum_j acty_kj wf2_j
  blas::gemm(1, knChains, kAlpha*knInputs, kone, kzero, &wf2_[0], &acty_[0], lnpsi);
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::backward(std::complex<FloatType> * lnpsiGradients)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size();
    for (int f=0; f<kAlpha; ++f)
    {
      d_dw1o_[kvSize+f] = kzero;
      d_db1_[kvSize+f] = kzero;
      for (int j=0; j<knInputs; ++j)
      {
        d_dw1o_[kvSize+f] += logcosh(y_[k*kAlpha*knInputs+f*knInputs+j]);
        d_db1_[kvSize+f] += wf2_[f*knInputs+j]*std::tanh(y_[k*kAlpha*knInputs+f*knInputs+j]);
      }
      for (int i=0; i<knInputs; ++i)
      {
        d_dwi1_[kvSize+f*knInputs+i] = kzero;
        for (int j=0; j<knInputs; ++j)
          d_dwi1_[kvSize+f*knInputs+i] += wf2_[f*knInputs+j]*
            std::tanh(y_[k*kAlpha*knInputs+f*knInputs+j])*
            spinStates_[k*knInputs+(knInputs+i-j)%knInputs].real();
      }
    }
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*knChains);
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
  if (rawdata.size() == variables_.size())
    variables_ = rawdata;
  else
    std::cout << " check parameter size... " << std::endl;
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::save(const std::string filePath, const int precision) const
{
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  for (const auto & var : variables_)
    writer << var << " ";
  writer.close();
}

template <typename FloatType>
void FFNNTrSymm<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    for (int j=0; j<kAlpha*knInputs; ++j)
      y_[k*kAlpha*knInputs+j] = y_[k*kAlpha*knInputs+j]-wf1_[index_*kAlpha*knInputs+j]*(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index_].real());
    spinStates_[k*knInputs+index_] = (kone.real()-ktwoTrueFalse[doSpinFlip[k]])*spinStates_[k*knInputs+index_].real();
  }
}


template <typename FloatType>
FFNNSfSymm<FloatType>::FFNNSfSymm(const int nInputs, const int alpha, const int nChains):
  knInputs(nInputs),
  kAlpha(alpha),
  knChains(nChains),
  kzero(std::complex<FloatType>(0.0, 0.0)),
  kone(std::complex<FloatType>(1.0, 0.0)),
  ktwo(2.0),
  koneChains(nChains, std::complex<FloatType>(1.0, 0.0)),
  ktwoTrueFalse({0.0, 2.0}),
  variables_(nInputs*alpha*nInputs + nInputs*alpha),
  lnpsiGradients_(nChains*(nInputs*alpha*nInputs + nInputs*alpha)),
  spinStates_(nInputs*nChains),
  y_(alpha*nInputs*nChains),
  acty_(alpha*nInputs*nChains),
  index_(0)
{
  unsigned long int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937_64 ran(seed);
  std::normal_distribution<double>
    randwi1(0, std::sqrt(1.0/((1+kAlpha)*knInputs))),
    randw1o(0, std::sqrt(1.0/(kAlpha*knInputs)));
  wi1_ = &variables_[0];
  w1o_ = &variables_[nInputs*kAlpha*knInputs];
  d_dwi1_ = &lnpsiGradients_[0];
  d_dw1o_ = &lnpsiGradients_[nInputs*kAlpha*knInputs];
  for (int i=0; i<nInputs*kAlpha*knInputs; ++i)
    wi1_[i] = std::complex<FloatType>(randwi1(ran), 1e-1*randwi1(ran));
  for (int j=0; j<kAlpha*knInputs; ++j)
    w1o_[j] = std::complex<FloatType>(randw1o(ran), 1e-1*randw1o(ran));
}

template <typename FloatType>
void FFNNSfSymm<FloatType>::update_variables(const std::complex<FloatType> * derivativeLoss, const FloatType learningRate)
{
  for (int i=0; i<variables_.size(); ++i)
    variables_[i] = variables_[i] - learningRate*derivativeLoss[i];
  // y_kj = \sum_i spinStates_ki w_ij + koneChains_k (x) b_j
  blas::gemm(kAlpha*knInputs, knChains, knInputs, kone, kzero, wi1_, &spinStates_[0], &y_[0]);
}

template <typename FloatType>
void FFNNSfSymm<FloatType>::initialize(std::complex<FloatType> * lnpsi, const std::complex<FloatType> * spinStates)
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
  blas::gemm(kAlpha*knInputs, knChains, knInputs, kone, kzero, wi1_, &spinStates_[0], &y_[0]);
  // acty_kj = ln(cosh(y_kj))
  #pragma omp parallel for
  for (int j=0; j<y_.size(); ++j)
    acty_[j] = logcosh(y_[j]);
  // lnpsi_k = \sum_j acty_kj w1o_j
  blas::gemm(1, knChains, kAlpha*knInputs, kone, kzero, w1o_, &acty_[0], lnpsi);
}

template <typename FloatType>
void FFNNSfSymm<FloatType>::forward(const int spinFlipIndex, std::complex<FloatType> * lnpsi)
{
  index_ = spinFlipIndex;
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
    for (int j=0; j<kAlpha*knInputs; ++j)
      acty_[k*kAlpha*knInputs+j] = logcosh(y_[k*kAlpha*knInputs+j]-wi1_[index_*kAlpha*knInputs+j]*(ktwo*spinStates_[k*knInputs+index_].real()));
  // lnpsi_k = \sum_j acty_kj w1o_j
  blas::gemm(1, knChains, kAlpha*knInputs, kone, kzero, w1o_, &acty_[0], lnpsi);
}

template <typename FloatType>
void FFNNSfSymm<FloatType>::backward(std::complex<FloatType> * lnpsiGradients)
{
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    const int kvSize = k*variables_.size(), kiSize = k*knInputs, khSize = k*kAlpha*knInputs;
    for (int j=0; j<kAlpha*knInputs; ++j)
      d_dw1o_[kvSize+j] = logcosh(y_[khSize+j]);
    for (int j=0; j<kAlpha*knInputs; ++j)
    {
      const std::complex<FloatType> tany_j = std::tanh(y_[khSize+j]);
      for (int i=0; i<knInputs; ++i)
        d_dwi1_[kvSize+i*kAlpha*knInputs+j] = tany_j*spinStates_[kiSize+i].real()*w1o_[j];
    }
  }
  std::memcpy(lnpsiGradients, &lnpsiGradients_[0], sizeof(std::complex<FloatType>)*variables_.size()*knChains);
}

template <typename FloatType>
void FFNNSfSymm<FloatType>::load(const std::string filePath)
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
  if (rawdata.size() == variables_.size())
    variables_ = rawdata;
  else
    std::cout << " check parameter size... " << std::endl;
}

template <typename FloatType>
void FFNNSfSymm<FloatType>::save(const std::string filePath, const int precision) const
{
  std::ofstream writer(filePath);
  writer << std::setprecision(precision);
  for (const auto & var : variables_)
    writer << var << " ";
  writer.close();
}

template <typename FloatType>
void FFNNSfSymm<FloatType>::spin_flip(const std::vector<bool> & doSpinFlip, const int spinFlipIndex)
{
  index_ = ((spinFlipIndex == -1) ? index_ : spinFlipIndex);
  #pragma omp parallel for
  for (int k=0; k<knChains; ++k)
  {
    for (int j=0; j<kAlpha*knInputs; ++j)
      y_[k*kAlpha*knInputs+j] = y_[k*kAlpha*knInputs+j]-wi1_[index_*kAlpha*knInputs+j]*(ktwoTrueFalse[doSpinFlip[k]]*spinStates_[k*knInputs+index_].real());
    spinStates_[k*knInputs+index_] = (kone.real()-ktwoTrueFalse[doSpinFlip[k]])*spinStates_[k*knInputs+index_].real();
  }
}
} // namespace spinhalf
