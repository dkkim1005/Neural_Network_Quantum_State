// Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

#pragma once

// List up all variational wave functions here...
enum class Ansatz {RBM_SH, FNN_SH};
namespace spinhalfsystem
{
template <typename FloatType> class ComplexRBM;
template <typename FloatType> class ComplexFNN;
} // namespace spinhalfsystem
template <Ansatz T, typename Property> struct AnsatzType_;
template <typename FloatType> struct AnsatzType_<Ansatz::RBM_SH, FloatType> { using Name = spinhalfsystem::ComplexRBM<FloatType>; };
template <typename FloatType> struct AnsatzType_<Ansatz::FNN_SH, FloatType> { using Name = spinhalfsystem::ComplexFNN<FloatType>; };
//
template <Ansatz T, typename Property>
struct AnsatzTraits
{
  using AnsatzType = typename AnsatzType_<T, Property>::Name;
  using FloatType = Property;
};
//
template <Ansatz T1, Ansatz T2, typename Property>
struct AnsatzeTraits
{
  using AnsatzType1 = typename AnsatzType_<T1, Property>::Name;
  using AnsatzType2 = typename AnsatzType_<T2, Property>::Name;
  using FloatType = Property;
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
