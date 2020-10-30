// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

// List up all variational wave functions here...
enum class Ansatz {RBM, RBMTrSymm, RBMSfSymm, FNN, FNNTrSymm, FNNSfSymm};
namespace spinhalf
{
template <typename FloatType> class ComplexRBM;
template <typename FloatType> class ComplexRBMTrSymm;
template <typename FloatType> class ComplexRBMSfSymm;
template <typename FloatType> class ComplexFNN;
template <typename FloatType> class ComplexFNNTrSymm;
template <typename FloatType> class ComplexFNNSfSymm;
} // namespace spinhalf
template <Ansatz T, typename Property> struct Ansatz_;
template <typename FloatType> struct Ansatz_<Ansatz::RBM, FloatType> { using T = spinhalf::ComplexRBM<FloatType>; };
template <typename FloatType> struct Ansatz_<Ansatz::RBMTrSymm, FloatType> { using T = spinhalf::ComplexRBMTrSymm<FloatType>; };
template <typename FloatType> struct Ansatz_<Ansatz::RBMSfSymm, FloatType> { using T = spinhalf::ComplexRBMSfSymm<FloatType>; };
template <typename FloatType> struct Ansatz_<Ansatz::FNN, FloatType> { using T = spinhalf::ComplexFNN<FloatType>; };
template <typename FloatType> struct Ansatz_<Ansatz::FNNTrSymm, FloatType> { using T = spinhalf::ComplexFNNTrSymm<FloatType>; };
template <typename FloatType> struct Ansatz_<Ansatz::FNNSfSymm, FloatType> { using T = spinhalf::ComplexFNNSfSymm<FloatType>; };
//
template <Ansatz T, typename FT>
struct AnsatzTraits
{
  using AnsatzType = typename Ansatz_<T, FT>::T;
  using FloatType = FT;
};
//
template <Ansatz T1, Ansatz T2, typename FT>
struct AnsatzeTraits
{
  using AnsatzType1 = typename Ansatz_<T1, FT>::T;
  using AnsatzType2 = typename Ansatz_<T2, FT>::T;
  using FloatType = FT;
};

// digits of the numeric precision w.r.t. FloatType
template <typename FloatType> struct FloatTypeTrait_;
template <> struct FloatTypeTrait_<float> { static constexpr int precision = 8; };
template <> struct FloatTypeTrait_<double> { static constexpr int precision = 15; };

// implementation for the circular list structure
template <typename FloatType = int>
class OneWayLinkedIndex
{
public:
  void set_item(const FloatType & item) { item_ = item; }
  void set_nextptr(OneWayLinkedIndex * nextPtr) { nextPtr_ = nextPtr; }
  OneWayLinkedIndex * next_ptr() const { return nextPtr_; }
  FloatType get_item() { return item_; }
private:
  FloatType item_;
  OneWayLinkedIndex * nextPtr_;
};

inline std::complex<float> logcosh(const std::complex<float> & z)
{
  const float x = z.real(), y = z.imag();
  const float absx = std::abs(x), cosy = std::cos(y), siny = std::sin(y);
  const float expabsm2x = std::exp(-2.0f*absx);
  const float real = (1.0f+expabsm2x)*cosy, imag = (1.0f-expabsm2x)*siny*std::copysign(1.0f, x);
  return std::log(std::complex<float>(real, imag))+(absx-0.6931472f);
}

inline std::complex<double> logcosh(const std::complex<double> & z)
{
  const double x = z.real(), y = z.imag();
  const double absx = std::abs(x), cosy = std::cos(y), siny = std::sin(y);
  const double expabsm2x = std::exp(-2.0*absx);
  const double real = (1.0+expabsm2x)*cosy, imag = (1.0-expabsm2x)*siny*std::copysign(1.0, x);
  return std::log(std::complex<double>(real, imag))+(absx-0.6931471805599453);
}
