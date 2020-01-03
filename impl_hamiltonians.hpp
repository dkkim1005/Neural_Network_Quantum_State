#pragma once

template <typename DerivedWFSampler, typename float_t>
BaseParallelVMC<DerivedWFSampler, float_t>::BaseParallelVMC(const int nMCUnitSteps, const int nChains,
const unsigned long seedDistance, const unsigned long seedNumber):
knMCUnitSteps(nMCUnitSteps),
knChains(nChains),
updateList_(nChains),
lnpsi0_(nChains),
lnpsi1_(nChains),
ratio_(nChains),
randDev_(nChains)
{
  // block splitting scheme for parallel Monte-Carlo
  for (int k=0; k<knChains; ++k)
  {
    randDev_[k].seed(seedNumber);
    randDev_[k].jump(2*seedDistance*k);
  }
}

template <typename DerivedWFSampler, typename float_t>
void BaseParallelVMC<DerivedWFSampler, float_t>::warm_up(const int nMCSteps)
{
  // memorize an initial state
  THIS_(DerivedWFSampler)->initialize(&lnpsi0_[0]);
  for (int k=0; k<knChains; ++k)
    updateList_[k] = true;
  THIS_(DerivedWFSampler)->accept_next_state(updateList_);
  // MCMC sampling for warming up
  this->do_mcmc_steps(nMCSteps);
}

template <typename DerivedWFSampler, typename float_t>
void BaseParallelVMC<DerivedWFSampler, float_t>::do_mcmc_steps(const int nMCSteps)
{
  // Markov chain MonteCarlo(MCMC) sampling with nskip iterations
  for (int n=0; n<(nMCSteps*knMCUnitSteps); ++n)
  {
    THIS_(DerivedWFSampler)->sampling(&lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<knChains; ++k)
      ratio_[k] = std::norm(std::exp(lnpsi1_[k]-lnpsi0_[k]));
    for (int k=0; k<knChains; ++k)
    {
      if (randUniform_(randDev_[k])<ratio_[k])
      {
        updateList_[k] = true;
        lnpsi0_[k] = lnpsi1_[k];
      }
      else
        updateList_[k] = false;
    }
    THIS_(DerivedWFSampler)->accept_next_state(updateList_);
  }
}

template <typename DerivedWFSampler, typename float_t>
void BaseParallelVMC<DerivedWFSampler, float_t>::get_htilda(std::complex<float_t> * htilda)
{
  THIS_(DerivedWFSampler)->get_htilda(htilda);
}

template <typename DerivedWFSampler, typename float_t>
void BaseParallelVMC<DerivedWFSampler, float_t>::get_lnpsiGradients(std::complex<float_t> * lnpsiGradients)
{
  THIS_(DerivedWFSampler)->get_lnpsiGradients(lnpsiGradients);
}

template <typename DerivedWFSampler, typename float_t>
void BaseParallelVMC<DerivedWFSampler, float_t>::evolve(const std::complex<float_t> * trueGradients, const float_t learningRate)
{
  THIS_(DerivedWFSampler)->evolve(trueGradients, learningRate);
}


template <typename float_t>
TFI_chain<float_t>::TFI_chain(ComplexRBM<float_t> & machine,
const float_t h, const float_t J, const unsigned long seedDistance, const unsigned long seedNumber):
BaseParallelVMC<TFI_chain<float_t>, float_t>(machine.get_nInputs(), machine.get_nChains(), seedDistance, seedNumber),
kh(h),
kJ(J),
kzero(0.0),
ktwo(2.0),
machine_(machine),
list_(machine.get_nInputs()),
diag_(machine.get_nChains()),
leftIdx_(machine.get_nInputs()),
rightIdx_(machine.get_nInputs())
{
  // (checker board update) list_ : 1,3,5,...,2,4,6,...
  const int nSites = machine.get_nInputs();
  for (int i=0; i<nSites; i++)
    list_[i].set_item(i);  
  int idx = 0;
  for (int i=2; i<nSites; i+=2)
  {
    list_[idx].set_nextptr(&list_[i]);
    idx = i;
  }
  for (int i=1; i<nSites; i+=2)
  {
    list_[idx].set_nextptr(&list_[i]);
    idx = i;
  }
  list_[idx].set_nextptr(&list_[0]);
  idxptr_ = &list_[0];
  // indexing of the nearest neighbor site(periodic boundary condition)
  for (int i=0; i<nSites; ++i)
  {
    leftIdx_[i] = ((i==0) ? nSites-1 : i-1);
    rightIdx_[i] = ((i==nSites-1) ? 0 : i+1);
  }
}

template <typename float_t>
void TFI_chain<float_t>::initialize(std::complex<float_t> * lnpsi)
{
  machine_.initialize(lnpsi);
  const std::complex<float_t> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  // diag_ = \sum_i spin_i*spin_{i+1}
  for (int k=0; k<nChains; ++k)
  {
    diag_[k] = kzero;
    for (int i=0; i<nSites-1; ++i)
      diag_[k] += spinPtr[k*nSites+i]*spinPtr[k*nSites+i+1];
    diag_[k] += spinPtr[k*nSites+nSites-1]*spinPtr[k*nSites];
  }
}

template <typename float_t>
void TFI_chain<float_t>::sampling(std::complex<float_t> * lnpsi)
{
  idxptr_ = idxptr_->next_ptr();
  machine_.forward(idxptr_->get_item(), lnpsi);
}

template <typename float_t>
void TFI_chain<float_t>::accept_next_state(const std::vector<bool> & updateList)
{
  const int idx = idxptr_->get_item();
  const std::complex<float_t> * spinPtr = machine_.get_spinStates();
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  for (int k=0; k<nChains; ++k)
  {
    if (updateList[k])
      diag_[k] -= ktwo*spinPtr[k*nSites+idx]*(spinPtr[k*nSites+leftIdx_[idx]]+spinPtr[k*nSites+rightIdx_[idx]]);
  }
  machine_.spin_flip(updateList);
}

template <typename float_t>
void TFI_chain<float_t>::get_htilda(std::complex<float_t> * htilda)
{
  /*
     htilda(s_0) = \sum_{s_1} <s_0|H|s_1>\frac{<s_1|psi>}{<s_0|psi>}
      --> J*diag + h*sum_i \frac{<(s_1, s_2,...,-s_i,...,s_n|psi>}{<(s_1, s_2,...,s_i,...,s_n|psi>}
   */
  const int nChains = machine_.get_nChains(), nSites = machine_.get_nInputs();
  for (int k=0; k<nChains; ++k)
    htilda[k] = kJ*diag_[k];
  for (int i=0; i<nSites; ++i)
  {
    machine_.forward(i, &lnpsi1_[0]);
    #pragma omp parallel for
    for (int k=0; k<nChains; ++k)
      htilda[k] += kh*std::exp(lnpsi1_[k] - lnpsi0_[k]);
  }
}

template <typename float_t>
void TFI_chain<float_t>::get_lnpsiGradients(std::complex<float_t> * lnpsiGradients)
{
  machine_.backward(lnpsiGradients);
}

template <typename float_t>
void TFI_chain<float_t>::evolve(const std::complex<float_t> * trueGradients, const float_t learningRate)
{
  machine_.update_variables(trueGradients, learningRate);
}
