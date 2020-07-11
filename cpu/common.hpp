// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

// List up all variational wave functions here...
enum class Ansatz {RBM, RBMSymm, FNN};
namespace spinhalf
{
template <typename FloatType> class ComplexRBM;
template <typename FloatType> class ComplexRBMSymm;
template <typename FloatType> class ComplexFNN;
} // namespace spinhalf
template <Ansatz T, typename Property> struct Ansatz_;
template <typename FloatType> struct Ansatz_<Ansatz::RBM, FloatType> { using T = spinhalf::ComplexRBM<FloatType>; };
template <typename FloatType> struct Ansatz_<Ansatz::RBMSymm, FloatType> { using T = spinhalf::ComplexRBMSymm<FloatType>; };
template <typename FloatType> struct Ansatz_<Ansatz::FNN, FloatType> { using T = spinhalf::ComplexFNN<FloatType>; };
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
