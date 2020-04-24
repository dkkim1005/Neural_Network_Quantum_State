// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <curand.h>
#include "common.cuh"

// RNGType : CURAND_RNG_PSEUDO_MT19937,...
template <typename FloatType, curandRngType RNGType>
class CurandWrapper 
{
public:
  explicit CurandWrapper(const unsigned long long seed);
  ~CurandWrapper();
  void get_uniformDist(double * data_dev, const int size);
private:
  curandGenerator_t rng_;
};

inline void curandGenerateUniformT(curandGenerator_t & rng, float * data_dev, const int size)
{
  CHECK_ERROR(CURAND_STATUS_SUCCESS, curandGenerateUniform(rng, data_dev, size));
}

inline void curandGenerateUniformT(curandGenerator_t & rng, double * data_dev, const int size)
{
  CHECK_ERROR(CURAND_STATUS_SUCCESS, curandGenerateUniformDouble(rng, data_dev, size));
}

template <typename FloatType, curandRngType RNGType>
CurandWrapper<FloatType, RNGType>::CurandWrapper(const unsigned long long seed)
{
  CHECK_ERROR(CURAND_STATUS_SUCCESS, curandCreateGenerator(&rng_, RNGType));
  CHECK_ERROR(CURAND_STATUS_SUCCESS, curandSetPseudoRandomGeneratorSeed(rng_, seed));
}

template <typename FloatType, curandRngType RNGType>
CurandWrapper<FloatType, RNGType>::~CurandWrapper()
{
  CHECK_ERROR(CURAND_STATUS_SUCCESS, curandDestroyGenerator(rng_));
}

template <typename FloatType, curandRngType RNGType>
void CurandWrapper<FloatType, RNGType>::get_uniformDist(double * data_dev, const int size)
{
  curandGenerateUniformT(rng_, data_dev, size);
}
