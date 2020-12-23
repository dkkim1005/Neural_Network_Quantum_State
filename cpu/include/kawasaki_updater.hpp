// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <iomanip>
#include <complex>

namespace kawasaki
{
// T <-- complex<float> or complex<double>
template <typename T>
struct IsBondState
{
  static bool eval(const T & c1, const T & c2)
  {
    return (c1.real()*c2.real() < 0);
  }
};

template <typename LatticeTraits>
class NNSpinExchanger
{
public:
  NNSpinExchanger(const int nChains, const int nInputs, const unsigned long seed0, const unsigned long seedJump):
    knChains(nChains),
    knInputs(nInputs),
    kmaxBondSize(nInputs*kNNeighbors/2),
    spinIdxTobondIdx_(nInputs),
    bondIdxTospinIdx_(nInputs*kNNeighbors/2),
    bondState_(nChains, std::vector<bool>(nInputs*kNNeighbors/2, false)),
    genVec_(nChains),
    nbond_(nChains),
    tmpbondIdx_(nChains),
    bondTable_(nChains, std::vector<int>(nInputs*kNNeighbors/2+1))
  {
    if (nInputs*kNNeighbors%2 == 1)
      throw std::invalid_argument("nInputs*kNNeighbors%2 == 1");
    unsigned long seed = seed0;
    for (auto & gen : genVec_)
    {
      gen.seed(seed);
      seed += seedJump;
    }
    // initialize 'spinIdxTobondIdx_' and 'bondIdxTospinIdx_';
    LatticeTraits::construct_spin_bond_indexing_rule(spinIdxTobondIdx_, bondIdxTospinIdx_);
  }

  // Bond states are generated with a given state (=data).
  template <typename EquivFunctor, typename FloatType>
  void init(const EquivFunctor & eq, const FloatType * data)
  {
    for (int k=0; k<knChains; ++k)
      for (int bondIdx=0; bondIdx<kmaxBondSize; ++bondIdx)
        bondState_[k][bondIdx] = eq.eval(data[k*knInputs+bondIdxTospinIdx_[bondIdx][0]],
                                         data[k*knInputs+bondIdxTospinIdx_[bondIdx][1]]);
    this->make_table_();
  }

  void get_indexes_of_spin_pairs(std::vector<std::vector<int> > & spinIdx)
  {
    for (int k=0; k<knChains; ++k)
    {
      tmpbondIdx_[k] = bondTable_[k][static_cast<int>(nbond_[k]*dist_(genVec_[k]))];
      spinIdx[k][0] = bondIdxTospinIdx_[tmpbondIdx_[k]][0];
      spinIdx[k][1] = bondIdxTospinIdx_[tmpbondIdx_[k]][1];
    }
  }

  void do_exchange(const std::vector<bool> & isExchanged)
  {
    for (int k=0; k<knChains; ++k)
    {
      const int booleanIdx = static_cast<int>(isExchanged[k]);
      // spinPairIdx: pairing index of a spin mediating a bond state
      for (int spinPairIdx=0; spinPairIdx<2; ++spinPairIdx)
        for (int n=0; n<kNNeighbors; ++n)
        {
          // nearest bond index of a spin index
          const int nnbondIdx = spinIdxTobondIdx_[bondIdxTospinIdx_[tmpbondIdx_[k]][spinPairIdx]][n];
          bondState_[k][nnbondIdx] = static_cast<bool>(booleanIdx+static_cast<int>(bondState_[k][nnbondIdx])*(1-2*booleanIdx));
        }
    }
    // arange table
    this->make_table_();
  }

private:
  void make_table_()
  {
    // The end index of 'bondTable_'(=EOT) is for the null data.
    const int EOT = bondTable_[0].size()-1;
    std::fill(nbond_.begin(), nbond_.end(), 0);
    for (int k=0; k<knChains; ++k)
      for (int bondIdx=0; bondIdx<kmaxBondSize; ++bondIdx)
      {
        const int booleanIdx = static_cast<int>(bondState_[k][bondIdx]);
        // Index as working a boolean operation on 'bondTable_'.
        const int booleanOpIdx = EOT+(nbond_[k]-EOT)*booleanIdx;
        // IF statement is replaced as following;
        // if (bondState_[k][bondIdx]) {bondTable_[k][nbond_[k]] = bondIdx; nbond_[k] += 1;} 
        // --> bondTable_[k][booleanOpIdx] = bondIdx; nbond_[k] += booleanIdx;
        bondTable_[k][booleanOpIdx] = bondIdx;
        nbond_[k] += booleanIdx;
      }
  }

  constexpr static int kNNeighbors = LatticeTraits::NNeighbors;
  const int knChains, knInputs, kmaxBondSize;
  std::vector<std::array<int, LatticeTraits::NNeighbors> > spinIdxTobondIdx_;
  std::vector<std::array<int, 2> > bondIdxTospinIdx_;
  // true : active bonding (antiferro)
  // false : inactive bonding (ferro)
  std::vector<std::vector<bool> > bondState_;
  std::vector<std::mt19937> genVec_;
  std::vector<int> nbond_, tmpbondIdx_;
  std::vector<std::vector<int> > bondTable_;
  std::uniform_real_distribution<double> dist_;
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
  static void construct_spin_bond_indexing_rule(std::vector<std::array<int, 2> > & spinIdxTobondIdx,
    std::vector<std::array<int, 2> > & bondIdxTospinIdx)
  {
    const int nInputs = spinIdxTobondIdx.size();
    for (int i=0; i<nInputs; ++i)
    {
      spinIdxTobondIdx[i][0] = ((i != 0) ? i-1 : nInputs-1);
      spinIdxTobondIdx[i][1] = i;
    }
    for (int bondIdx=0; bondIdx<nInputs; ++bondIdx)
    {
      bondIdxTospinIdx[bondIdx][0] = bondIdx;
      bondIdxTospinIdx[bondIdx][1] = (bondIdx != nInputs-1) ? bondIdx+1 : 0;
    }
  }
};

typedef NNSpinExchanger<ChainLattice> NNSpinExchanger1D;
} // end namespace kawasaki
