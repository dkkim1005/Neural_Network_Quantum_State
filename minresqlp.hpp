// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

/*
 MINRESQLP C++ implementation based on the original fortran90 code, minresqlpModule.f90(09 Sep 2013)

  * Original author and contributor:
    Author:
      Sou-Cheng Choi <sctchoi@uchicago.edu>
      Computation Institute (CI)
      University of Chicago
      Chicago, IL 60637, USA

      Michael Saunders <saunders@stanford.edu>
      Systems Optimization Laboratory (SOL)
      Stanford University
      Stanford, CA 94305-4026, USA

    Contributor:
      Christopher Paige <paige@cs.mcgill.ca>

  * C++ developer:
    Dongkyu Kim <dkkim1005@gmail.com>
    Computational Many-body Physics Group
    Gwangju Institute of Science and Technology(GIST)
    Gwangju, South korea

  * Date:
    ver 1.0 / 27 april 2018
     - first realise
    ver 1.1 / 16 january 2020
     - CRTP design is applied to BaseInterface class.
     - extend supports of float type of BaseInterface class

  Searching for detailed descriptions, see http://web.stanford.edu/group/SOL/software/minresqlp/
*/


#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <complex>
#include <algorithm>
#include <numeric>
#include <iomanip>

// NumberType
template <typename FloatType>
using REAL = FloatType;
template <typename FloatType>
using IMAG = std::complex<FloatType>;

/*
 * NumberTypeFloat:
 * ex) REAL<float> --> float, Imag<double> --> double
 */
template <typename NumberType> struct NumberTypeTrait_ {};
template <> struct NumberTypeTrait_<REAL<float> > { typedef float FloatType; };
template <> struct NumberTypeTrait_<REAL<double> > { typedef double FloatType; };
template <> struct NumberTypeTrait_<IMAG<float> > { typedef float FloatType; };
template <> struct NumberTypeTrait_<IMAG<double> > { typedef double FloatType; };
template <typename NumberType>
using NumberTypeFloat = typename NumberTypeTrait_<NumberType>::FloatType;

namespace MINRESQLP
{
/*
 * Base class for the interface of MINRESQLP
 * CRTP design is applied to here for static polymorphism;
 * only remained tasks are implementing DerivedOP class which has Aprod & Msolve(optional) methods.
 */
template<typename DerivedOP, typename NumberType>
class BaseInterface
{
public:
  typedef std::vector<NumberType> ContainerType;
  BaseInterface(const int n_,
    const NumberType * b_,
    const NumberTypeFloat<NumberType> shift_ = 0,
    const bool useMsolve_ = false,
    const bool disable_ = false,
    const int itnlim_ = -1,
    const NumberTypeFloat<NumberType> rtol_ = 1e-16,
    const NumberTypeFloat<NumberType> maxxnorm_ = 1e7,
    const NumberTypeFloat<NumberType> trancond_ = 1e7,
    const NumberTypeFloat<NumberType> Acondlim_ = 1e15,
    const bool print_ = false);
  void Aprod(const int n, const NumberType *x, NumberType *y) const;
  void Msolve(const int n, const NumberType *x, NumberType *y) const;
  // inputs
  int n, itnlim;
  ContainerType b;
  NumberTypeFloat<NumberType> shift, rtol, maxxnorm, trancond, Acondlim;
  bool useMsolve, disable, print;
  // outputs
  ContainerType x;
  int istop, itn;
  NumberTypeFloat<NumberType> rnorm, Arnorm, xnorm, Anorm, Acond;
};


template<typename DerivedOP, typename FloatType>
class RealSolver
{
public:
  void solve(BaseInterface<DerivedOP, REAL<FloatType> > & client) const;
private:
  void symortho_(const FloatType& a, const FloatType& b, FloatType &c, FloatType &s, FloatType &r) const;
  FloatType dnrm2_(const int n, const FloatType* x, const int incx) const;
  void printstate_(const int iter, const FloatType x1, const FloatType xnorm,
    const FloatType rnorm, const FloatType Arnorm, const FloatType relres,
    const FloatType relAres, const FloatType Anorm, const FloatType Acond) const;
  static constexpr FloatType eps_ = std::numeric_limits<FloatType>::epsilon();
};


template<typename DerivedOP, typename FloatType>
class HermitianSolver
{
public:
  void solve(BaseInterface<DerivedOP, IMAG<FloatType> > & client) const;
private:
  void zsymortho_(const std::complex<FloatType>& a, const std::complex<FloatType>& b,
    FloatType& c, std::complex<FloatType>& s, std::complex<FloatType>& r) const;
  std::complex<FloatType> zdotc_(const int n, const std::complex<FloatType>* cx,
    const int incx, const std::complex<FloatType>* cy, const int incy) const;
  FloatType znrm2_(const int n, const std::complex<FloatType>* x, const int incx) const;
  void printstate_(const int iter, const std::complex<FloatType> x1, const FloatType xnorm,
    const FloatType rnorm, const FloatType Arnorm, const FloatType relres,
    const FloatType relAres, const FloatType Anorm, const FloatType Acond) const;
  static constexpr FloatType eps_ = std::numeric_limits<FloatType>::epsilon();
  static constexpr FloatType realmin_ = std::numeric_limits<FloatType>::min();
};

} // end namespace MINRESQLP

#include "impl_minresqlp.hpp"
