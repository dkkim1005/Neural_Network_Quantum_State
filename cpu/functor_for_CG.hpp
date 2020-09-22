#include <vector>
#include <complex>
#include "blas_lapack.hpp"

template <typename FloatType>
class SMatrixForCG
{
public:
  SMatrixForCG(const int nChains, const int nVariables);
  // (lambda): S = S + \lambda*diag(S)
  void set_lnpsiGradients(const std::complex<FloatType> * lnpsiGradients, const FloatType lambda = 0);
  // b = S*a
  void dot(const std::complex<FloatType> * a, std::complex<FloatType> * b);
  // solve M*a = b where M is the preconditioner
  void applyPrecond(const std::complex<FloatType> * rhs, std::complex<FloatType> * x) const;

private:
  std::complex<FloatType> inner_product_(const std::complex<FloatType> * z1,
    const std::complex<FloatType> * z2, const int size) const;

  const int knChains, knVariables;
  const std::complex<FloatType> kone, kzero, kmone, kinvNchains;
  const std::vector<std::complex<FloatType> > kones;
  const std::complex<FloatType> * lnpsiGradientsPtr_;
  std::vector<std::complex<FloatType> > avglnpsiGradients_, z_;
  std::vector<FloatType> diag_;
  FloatType lambda_;
};

template <typename FloatType>
SMatrixForCG<FloatType>::SMatrixForCG(const int nChains, const int nVariables):
  knChains(nChains),
  knVariables(nVariables),
  kone(1, 0),
  kzero(0, 0),
  kmone(-1, 0),
  kinvNchains(static_cast<FloatType>(1)/nChains, static_cast<FloatType>(0)),
  kones(nChains, 1),
  avglnpsiGradients_(nVariables, 0),
  diag_(nVariables, 0),
  z_(nChains),
  lambda_(0) {}

template <typename FloatType>
void SMatrixForCG<FloatType>::set_lnpsiGradients(const std::complex<FloatType> * lnpsiGradients, const FloatType lambda)
{
  lnpsiGradientsPtr_ = lnpsiGradients;
  lambda_ = lambda;
  // avglnpsiGradients_{i} = 1/nChains*\sum_{k}(lnpsiGradients_{k,i})
  blas::gemv(knVariables, knChains, kinvNchains, lnpsiGradientsPtr_, kones.data(), kzero, avglnpsiGradients_.data());
  // diag_{i} = 1/nChains*\sum_{k}(|lnpsiGradients_{k,i}|^2) - |avglnpsiGradients_{i}|^2
  const FloatType invNsamples = 1/static_cast<FloatType>(knChains);
  std::fill(diag_.begin(), diag_.end(), 0);
  for (int i=0; i<knVariables; ++i)
  {
    for (int k=0; k<knChains; ++k)
      diag_[i] += std::norm(lnpsiGradients[k*knVariables+i]);
    diag_[i] = invNsamples*diag_[i]-std::norm(avglnpsiGradients_[i]);
  }
}

template <typename FloatType>
void SMatrixForCG<FloatType>::dot(const std::complex<FloatType> * a, std::complex<FloatType> * b)
{
  // z_{k} = \sum_{i}(lnpsiGradients_{k,i}*a_{i})
  blas::gemm(1, knChains, knVariables, kone, kzero, a, lnpsiGradientsPtr_, &z_[0]);
  // z = conj(z)
  for (int i=0; i<knChains; ++i)
    z_[i] = std::conj(z_[i]);
  // tmp = conj(\sum_{i}(avglnpsiGradients_{i}*a_{i}))
  const std::complex<FloatType> tmp = std::conj(this->inner_product_(&avglnpsiGradients_[0], &a[0], knVariables));
  // b_{i} = avglnpsiGradients_{i}*tmp => \sum_{j}(avglnpsiGradients_{i}*conj(avglnpsiGradients_{j})*conj(a_{j}))
  for (int j=0; j<knVariables; ++j)
    b[j] = tmp*avglnpsiGradients_[j];
  // b_{i} = 1/nChains*\sum_{k}(lnpsiGradients_{k,i}*z_{k}) - b_{i} 
  blas::gemv(knVariables, knChains, kinvNchains, lnpsiGradientsPtr_, &z_[0], kmone, b);
  // conj(b) => (<var*var>-<var>*<var>)*a
  for (int j=0; j<knVariables; ++j)
    b[j] = std::conj(b[j]);
  // b_{i} = lambda*a_{i}*diag_{i}+b_{i}
  for (int i=0; i<knVariables; ++i)
    b[i] += lambda_*diag_[i]*a[i];
}

template <typename FloatType>
void SMatrixForCG<FloatType>::applyPrecond(const std::complex<FloatType> * rhs, std::complex<FloatType> * x) const
{
  // Diagonal preconditioner
  for (int i=0; i<knVariables; ++i)
    x[i] = rhs[i]/((1+lambda_)*diag_[i]);
}

template <typename FloatType>
std::complex<FloatType> SMatrixForCG<FloatType>::inner_product_(const std::complex<FloatType> * z1,
  const std::complex<FloatType> * z2, const int size) const
{
  std::complex<FloatType> res = 0;
  for (int i=0; i<size; ++i)
    res += z1[i]*z2[i];
  return res;
}
