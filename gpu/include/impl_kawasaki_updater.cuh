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
void NNSpinExchanger<LatticeTraits, RNGFloatType>::get_indexes_of_spin_pairs(thrust::device_vector<int> & spinIdx_dev)
{
  rng_.get_uniformDist(PTR_FROM_THRUST(rngValues_dev_.data()));
  gpu_kernel::NNSpinExchanger__AssignSpinPairs__<<<kgpuBlockSize, NUM_THREADS_PER_BLOCK >>>
    (knChains, kmaxBondSize,
     PTR_FROM_THRUST(bondTable_dev_.data()),
     PTR_FROM_THRUST(rngValues_dev_.data()),
     PTR_FROM_THRUST(bondIdxTospinIdx_dev_.data()), 
     PTR_FROM_THRUST(nbond_dev_.data()),
     PTR_FROM_THRUST(tmpbondIdx_dev_.data()),
     PTR_FROM_THRUST(spinIdx_dev.data()));
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
  const int * nbond, int * tmpbondIdx, int * spinIdx)
{
  const unsigned int nstep = gridDim.x*blockDim.x;
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  while (idx < nChains)
  {
    tmpbondIdx[idx] = bondTable[idx*(maxBondSize+1)+static_cast<int>(nbond[idx]*rngValues[idx])];
    spinIdx[2*idx+0] = bondIdxTospinIdx[2*tmpbondIdx[idx]+0];
    spinIdx[2*idx+1] = bondIdxTospinIdx[2*tmpbondIdx[idx]+1];
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
    const int booleanIdx = static_cast<int>(isExchanged[idx]);
    // spinPairIdx: pairing index of a spin mediating a bond state
    for (int spinPairIdx=0; spinPairIdx<2; ++spinPairIdx)
      for (int n=0; n<NNeighbors; ++n)
      {
        // nearest bond index of a spin index
        const int nnbondIdx = spinIdxTobondIdx[NNeighbors*bondIdxTospinIdx[2*tmpbondIdx[idx]+spinPairIdx]+n];
        bondState[maxBondSize*idx+nnbondIdx] = static_cast<bool>(booleanIdx+static_cast<int>(bondState[maxBondSize*idx+nnbondIdx])*(1-2*booleanIdx));
      }
    idx += nstep;
  }
}
} // end namespace gpu_kernel
