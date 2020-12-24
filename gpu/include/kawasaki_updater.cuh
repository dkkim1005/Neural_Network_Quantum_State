// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include "trng4cuda.cuh"
#include "common.cuh"

namespace gpu_kernel
{
__global__ void NNSpinExchanger__MakeTable__(const int nChains, const int maxBondSize,
  const bool * bondState, int * nbond, int * bondTable);

template <typename FloatType>
__global__ void NNSpinExchanger__AssignSpinPairs__(const int nChains, const int maxBondSize,
  const int * bondTable, const FloatType * rngValues, const int * bondIdxTospinIdx,
  const int * nbond, int * tmpbondIdx, int * spinIdx);

template <int NNeighbors>
__global__ void NNSpinExchanger__UpdateBondState__(const int nChains, const int maxBondSize,
  const bool * isExchanged, const int * spinIdxTobondIdx, const int * bondIdxTospinIdx,
  const int * tmpbondIdx, bool * bondState);
} // end namespace gpu_kernel

namespace kawasaki
{
struct IsBondState
{
  template <typename T>
  static bool eval(const T & s1, const T & s2)
  {
    return (s1.real()*s2.real() < 0);
  }
};

template <typename LatticeTraits, typename RNGFloatType = double>
class NNSpinExchanger
{
public:
  NNSpinExchanger(const int nChains, const int nInputs, const unsigned long seed0, const unsigned long seedJump);
  // Bond states are generated with a given state (=data).
  template <typename EquivFunctor, typename T>
  void init(const EquivFunctor & eq, const thrust::device_ptr<T> & data_ptr_dev)
  {
    thrust::host_vector<T> data_host(data_ptr_dev, data_ptr_dev+knChains*knInputs);
    thrust::host_vector<bool> bondState_host(bondState_dev_.size());
    thrust::host_vector<int> bondIdxTospinIdx_host(bondIdxTospinIdx_dev_);
    for (int k=0; k<knChains; ++k)
      for (int bondIdx=0; bondIdx<kmaxBondSize; ++bondIdx)
      {
        const int leftIdx = k*knInputs+bondIdxTospinIdx_host[2*bondIdx+0],
          rightIdx = k*knInputs+bondIdxTospinIdx_host[2*bondIdx+1];
        // check whether bond state is constructed
        bondState_host[k*kmaxBondSize+bondIdx] = eq.eval(data_host[leftIdx], data_host[rightIdx]);
      }
    bondState_dev_ = bondState_host;
    gpu_kernel::NNSpinExchanger__MakeTable__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>(knChains, kmaxBondSize,
      PTR_FROM_THRUST(bondState_dev_.data()),
      PTR_FROM_THRUST(nbond_dev_.data()),
      PTR_FROM_THRUST(bondTable_dev_.data()));
  }
  void get_indexes_of_spin_pairs(thrust::device_vector<int> & spinIdx_dev);
  void do_exchange(const bool * isExchanged_dev);
private:
  constexpr static int kNNeighbors = LatticeTraits::NNeighbors;
  const int knChains, knInputs, kmaxBondSize, kgpuBlockSize;
  thrust::device_vector<int> spinIdxTobondIdx_dev_;
  thrust::device_vector<int> bondIdxTospinIdx_dev_;
  // true : active bonding (antiferro)
  // false : inactive bonding (ferro)
  thrust::device_vector<bool> bondState_dev_;
  thrust::device_vector<int> nbond_dev_, tmpbondIdx_dev_;
  thrust::device_vector<int> bondTable_dev_;
  TRNGWrapper<RNGFloatType, trng::yarn2> rng_;
  thrust::device_vector<RNGFloatType> rngValues_dev_;
};

struct ChainLattice
{
  //  schematic diagram for spin-bond structure
  // O : spin / (x) : bond
  // 0           1          2          3              N-1
  // O -- (x) -- O -- (x) --O -- (x) --O ... -- (x) -- O
  // |     0           1          2             N-2    |
  // |----------------------- (x) ---------------------|
  //                          N-1
  constexpr static int NNeighbors = 2;
  static void construct_spin_bond_indexing_rule(thrust::device_vector<int> & spinIdxTobondIdx_dev,
    thrust::device_vector<int> & bondIdxTospinIdx_dev)
  {
    const int nInputs = spinIdxTobondIdx_dev.size()/NNeighbors;
    thrust::host_vector<int> spinIdxTobondIdx_host(spinIdxTobondIdx_dev.size()),
      bondIdxTospinIdx_host(bondIdxTospinIdx_dev.size());
    for (int i=0; i<nInputs; ++i)
    {
      spinIdxTobondIdx_host[2*i+0] = ((i != 0) ? i-1 : nInputs-1);
      spinIdxTobondIdx_host[2*i+1] = i;
    }
    for (int bondIdx=0; bondIdx<nInputs; ++bondIdx)
    {
      bondIdxTospinIdx_host[2*bondIdx+0] = bondIdx;
      bondIdxTospinIdx_host[2*bondIdx+1] = (bondIdx != nInputs-1) ? bondIdx+1 : 0;
    }
    spinIdxTobondIdx_dev = spinIdxTobondIdx_host;
    bondIdxTospinIdx_dev = bondIdxTospinIdx_host;
  }
};

template <typename RNGFloatType>
using NNSpinExchanger1D = NNSpinExchanger<ChainLattice, RNGFloatType>;
} // end namespace kawasaki 

#include "impl_kawasaki_updater.cuh"
