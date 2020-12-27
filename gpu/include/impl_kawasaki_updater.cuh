// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

namespace kawasaki
{
template <typename LatticeTraits, typename RNGFloatType>
NNSpinExchanger<LatticeTraits, RNGFloatType>::NNSpinExchanger(const int nChains,
  const int nInputs, const unsigned long seed0, const unsigned long seedJump):
  knChains(nChains),
  knInputs(nInputs),
  kmaxBondSize(nInputs*kNNeighbors/2),
  // spin index --> bond index
  spinIdxTobondIdx_dev_(nInputs*kNNeighbors),
  // bond index --> spin index
  bondIdxTospinIdx_dev_(nInputs*kNNeighbors),
  bondState_dev_(nChains*nInputs*kNNeighbors/2, false),
  nbond_dev_(nChains),
  tmpbondIdx_dev_(nChains),
  bondTable_dev_(nChains*(nInputs*kNNeighbors/2+1)),
  kgpuBlockSize(1+(nChains-1)/NUM_THREADS_PER_BLOCK),
  rng_(seed0, seedJump, nChains),
  rngValues_dev_(nChains)
{
  if (nInputs*kNNeighbors%2 == 1)
    throw std::invalid_argument("nInputs*kNNeighbors%2 == 1");
  // initialize 'spinIdxTobondIdx_' and 'bondIdxTospinIdx_';
  LatticeTraits::construct_spin_bond_indexing_rule(spinIdxTobondIdx_dev_, bondIdxTospinIdx_dev_);
}

template <typename LatticeTraits, typename RNGFloatType>
void NNSpinExchanger<LatticeTraits, RNGFloatType>::get_indexes_of_spin_pairs(thrust::device_vector<thrust::pair<int, int> > & spinPairIdx_dev)
{
  rng_.get_uniformDist(PTR_FROM_THRUST(rngValues_dev_.data()));
  gpu_kernel::NNSpinExchanger__AssignSpinPairs__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK >>>
    (knChains, kmaxBondSize,
     PTR_FROM_THRUST(bondTable_dev_.data()),
     PTR_FROM_THRUST(rngValues_dev_.data()),
     PTR_FROM_THRUST(bondIdxTospinIdx_dev_.data()), 
     PTR_FROM_THRUST(nbond_dev_.data()),
     PTR_FROM_THRUST(tmpbondIdx_dev_.data()),
     PTR_FROM_THRUST(spinPairIdx_dev.data()));
}

template <typename LatticeTraits, typename RNGFloatType>
void NNSpinExchanger<LatticeTraits, RNGFloatType>::do_exchange(const bool * isExchanged_dev)
{
  gpu_kernel::NNSpinExchanger__UpdateBondState__<kNNeighbors><<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>
    (knChains, kmaxBondSize,
     isExchanged_dev,
     PTR_FROM_THRUST(spinIdxTobondIdx_dev_.data()),
     PTR_FROM_THRUST(bondIdxTospinIdx_dev_.data()),
     PTR_FROM_THRUST(tmpbondIdx_dev_.data()),
     PTR_FROM_THRUST(bondState_dev_.data()));
  // arange table
  gpu_kernel::NNSpinExchanger__MakeTable__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK>>>
    (knChains, kmaxBondSize,
     PTR_FROM_THRUST(bondState_dev_.data()),
     PTR_FROM_THRUST(nbond_dev_.data()),
     PTR_FROM_THRUST(bondTable_dev_.data()));
}
} // end namespace kawasaki 


namespace gpu_kernel
{
// The end index of 'bondTable'(=EOT) is for the null data.
__global__ void NNSpinExchanger__MakeTable__(const int nChains, const int maxBondSize,
  const bool * bondState, int * nbond, int * bondTable)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  const int EOT = maxBondSize;
  while (idx < nChains)
  {
    nbond[idx] = 0;
    for (int bondIdx=0; bondIdx<maxBondSize; ++bondIdx)
    {
      const int booleanIdx = static_cast<int>(bondState[idx*maxBondSize+bondIdx]);
      // Index as working a boolean operation on 'bondTable'.
      const int booleanOpIdx = EOT+(nbond[idx]-EOT)*booleanIdx;
      // IF statement is replaced as following;
      // if (bondState[idx][bondIdx]) {bondTable[idx][nbond[idx]] = bondIdx; nbond[idx] += 1;} 
      // --> bondTable[idx][booleanOpIdx] = bondIdx; nbond[idx] += booleanIdx;
      bondTable[idx*(maxBondSize+1)+booleanOpIdx] = bondIdx;
      nbond[idx] += booleanIdx;
    }
    idx += nstep;
  }
}

template <typename FloatType>
__global__ void NNSpinExchanger__AssignSpinPairs__(const int nChains, const int maxBondSize,
  const int * bondTable, const FloatType * rngValues, const int * bondIdxTospinIdx,
  const int * nbond, int * tmpbondIdx, thrust::pair<int, int> * spinPairIdx)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    tmpbondIdx[idx] = bondTable[idx*(maxBondSize+1)+static_cast<int>(nbond[idx]*rngValues[idx])];
    spinPairIdx[idx].first = bondIdxTospinIdx[2*tmpbondIdx[idx]+0];
    spinPairIdx[idx].second = bondIdxTospinIdx[2*tmpbondIdx[idx]+1];
    idx += nstep;
  }
}

template <int NNeighbors>
__global__ void NNSpinExchanger__UpdateBondState__(const int nChains, const int maxBondSize,
  const bool * isExchanged, const int * spinIdxTobondIdx, const int * bondIdxTospinIdx,
  const int * tmpbondIdx, bool * bondState)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    const int boolIdx = static_cast<int>(isExchanged[idx]);
    // sp : pairing index of a spin mediating a bond state
    for (int sp=0; sp<2; ++sp)
      // n : index of the nearest neighbor of a spin
      for (int n=0; n<NNeighbors; ++n)
      {
        // nearest bond index of a spin index
        const int nbIdx = spinIdxTobondIdx[NNeighbors*bondIdxTospinIdx[2*tmpbondIdx[idx]+sp]+n];
        bondState[maxBondSize*idx+nbIdx] = static_cast<bool>(boolIdx+static_cast<int>(bondState[maxBondSize*idx+nbIdx])*(1-2*boolIdx));
      }
    idx += nstep;
  }
}
} // end namespace gpu_kernel
