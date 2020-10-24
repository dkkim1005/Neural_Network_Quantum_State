// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <iomanip>

namespace kawasaki
{
class NNSpinExchanger1D
{
  //  schematic diagram for spin-bond structure (chain lattice)
  // O : spin / (x) : bond
  // 0           1          2          3              N-1
  // O -- (x) -- O -- (x) --O -- (x) --O ... -- (x) -- O
  // |     0           1          2             N-2    |
  // |----------------------- (x) ---------------------|
  //                          N-1
public:
  NNSpinExchanger1D(const int nChains, const int nInputs, const unsigned long seed0, const unsigned long seedJump):
    knChains(nChains),
    knInputs(nInputs),
    SiteIndexes_(nInputs),
    BondState_(nChains, std::vector<bool>(nInputs, false)),
    NNBondIndexes_(nInputs),
    table_(nChains, std::vector<int>(nInputs)),
    tableSize_(nChains, 0),
    tmpbi_(nChains),
    genVec_(nChains)
  {
    for (int i=0; i<knInputs; ++i)
    {
      SiteIndexes_[i][0] = i;
      SiteIndexes_[i][1] = ((i!=knInputs-1) ? (i+1) : 0);
      NNBondIndexes_[i][0] = ((i!=0) ? i-1 : knInputs-1);
      NNBondIndexes_[i][1] = ((i!=knInputs-1) ? (i+1) : 0);
    }
    for (int ui = 0; ui < nChains; ++ui)
    {
      const unsigned long seed = seed0+static_cast<unsigned long>(ui)*seedJump;
      genVec_[ui].seed(seed);
    }
  }

  template <typename EquivFunctor, typename FloatType>
  void init(const EquivFunctor & eq, const FloatType * data)
  {
    for (int k=0; k<knChains; ++k)
      for (int bi=0; bi<knInputs; ++bi)
        BondState_[k][bi] = eq.eval(data[k*knInputs+SiteIndexes_[bi][0]], data[k*knInputs+SiteIndexes_[bi][1]]);
    this->make_table_();
  }

  void get_indexes_of_spin_pairs(std::vector<std::vector<int> > & idx)
  {
    for (int k=0; k<knChains; ++k)
    {
      tmpbi_[k] = table_[k][static_cast<int>(tableSize_[k]*dist_(genVec_[k]))];
      idx[k][0] = SiteIndexes_[tmpbi_[k]][0];
      idx[k][1] = SiteIndexes_[tmpbi_[k]][1];
    }
  }

  void do_exchange(const std::vector<bool> & state)
  {
    for (int k=0; k<knChains; ++k)
    {
      if (state[k])
      {
        BondState_[k][NNBondIndexes_[tmpbi_[k]][0]] = !BondState_[k][NNBondIndexes_[tmpbi_[k]][0]];
        BondState_[k][NNBondIndexes_[tmpbi_[k]][1]] = !BondState_[k][NNBondIndexes_[tmpbi_[k]][1]];
        // resize table
	tableSize_[k] = 0;
        for (int bi=0; bi<knInputs; ++bi)
          if (BondState_[k][bi])
	  {
            table_[k][tableSize_[k]] = bi;
            tableSize_[k] += 1;
	  }
      }
    }
  }

private:
  void make_table_()
  {
    std::fill(tableSize_.begin(), tableSize_.end(), 0);
    for (int k=0; k<knChains; ++k)
      for (int bi=0; bi<knInputs; ++bi)
        if (BondState_[k][bi])
	{
          table_[k][tableSize_[k]] = bi;
          tableSize_[k] += 1;
	}
  }

  const int knChains, knInputs;
  // bond_index : 0 ~ knInputs-1
  // obj[bond_index][0] : left site index
  // obj[bond_index][1] : right site index
  std::vector<std::array<int, 2> > SiteIndexes_;
  // true : active bonding (antiferro)
  // false : inactive bonding (ferro)
  std::vector<std::vector<bool> > BondState_;
  // obj[bond_index][0] : left bond index
  // obj[bond_index][1] : right bond index
  std::vector<std::array<int, 2> > NNBondIndexes_;
  std::vector<std::vector<int> > table_;
  std::vector<int> tableSize_, tmpbi_;
  std::vector<std::mt19937> genVec_;
  std::uniform_real_distribution<double> dist_;
};

template <typename T>
struct Equivfunc
{
  static bool eval(const T & c1, const T & c2)
  {
    return (std::norm(c1-c2) > 0);
  }
};
} // end namespace kawasaki
