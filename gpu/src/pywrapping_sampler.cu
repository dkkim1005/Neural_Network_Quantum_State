#include <memory>
#include "../include/neural_quantum_state.cuh"
#include "../include/meas.cuh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/numpy.h"

#define MAKE_PYSAMPLER_MODULE(MODULE_NAME, PYCLASS_NAME, ANSATZ, TYPE) do {\
  py::class_<PySampler<ANSATZ, TYPE>>(MODULE_NAME, PYCLASS_NAME)\
    .def(py::init<const py::dict>())\
    .def("load", &PySampler<ANSATZ, TYPE>::load)\
    .def("warm_up", &PySampler<ANSATZ, TYPE>::warm_up)\
    .def("do_mcmc_steps", &PySampler<ANSATZ, TYPE>::do_mcmc_steps)\
    .def("get_spinStates", &PySampler<ANSATZ, TYPE>::get_spinStates)\
    .def("get_lnpsi", &PySampler<ANSATZ, TYPE>::get_lnpsi)\
    .def("get_lnpsi_for_fixed_spins", &PySampler<ANSATZ, TYPE>::get_lnpsi_for_fixed_spins);\
} while (false)

namespace py = pybind11;

template <template<typename> class ToAnsatzType, typename ToFloatType>
struct Traits
{
  using AnsatzType = ToAnsatzType<ToFloatType>;
  using FloatType = ToFloatType;
};

template <template<typename> class ansatz, typename T>
class PySampler
{
  using traits = Traits<ansatz, T>;
  struct ComplexToReal
  { static T transform(const thrust::complex<T> & rhs) { return rhs.real(); } };
  struct ComplexToComplex
  { static std::complex<T> transform(const thrust::complex<T> & rhs) { return rhs; } };
public:
  PySampler(const py::dict kwargs)
  {
    const auto nInputs = kwargs["nInputs"].cast<int>(),
      nHiddens = kwargs["nHiddens"].cast<int>(),
      nChains = kwargs["nChains"].cast<int>();
    const auto seedNumber = kwargs["seedNumber"].cast<unsigned long>(),
      seedDistance = kwargs["seedDistance"].cast<unsigned long>();
    nqs_ptr0_ = std::make_unique<typename traits::AnsatzType>(nInputs, nHiddens, nChains);
    nqs_ptr1_ = std::make_unique<typename traits::AnsatzType>(nInputs, nHiddens, nChains);
    sampler_ptr_ = std::make_unique<Sampler4SpinHalf<traits>>(*nqs_ptr0_, seedNumber, seedDistance);
    spinStates_host_.resize(nInputs*nChains);
    spinStates_dev_.resize(nInputs*nChains);
    lnpsi_host_.resize(nChains);
    lnpsi_dev_.resize(nChains);
  }

  void load(const std::string prefix) const
  {
    nqs_ptr0_->load(prefix);
    nqs_ptr0_->copy_to(*nqs_ptr1_);
  }

  void warm_up(const int nMCSteps) const
  {
    sampler_ptr_->warm_up(nMCSteps);
  }

  void do_mcmc_steps(const int nMCSteps) const
  {
    sampler_ptr_->do_mcmc_steps(nMCSteps);
  }

  py::array_t<T> get_spinStates()
  {
    CHECK_ERROR(cudaSuccess, cudaMemcpy(spinStates_host_.data(),
      nqs_ptr0_->get_spinStates(),
      sizeof(thrust::complex<T>)*spinStates_host_.size(),
      cudaMemcpyDeviceToHost));
    return this->make_array_<thrust::complex<T>, T, ComplexToReal>(spinStates_host_);
  }

  py::array_t<std::complex<T>> get_lnpsi()
  {
    CHECK_ERROR(cudaSuccess, cudaMemcpy(lnpsi_host_.data(),
      sampler_ptr_->get_lnpsi(),
      sizeof(thrust::complex<T>)*lnpsi_host_.size(),
      cudaMemcpyDeviceToHost));
    return this->make_array_<thrust::complex<T>, std::complex<T>, ComplexToComplex>(lnpsi_host_);
  }

  py::array_t<std::complex<T>> get_lnpsi_for_fixed_spins(const py::array_t<T> spinStates_py)
  {
    py::buffer_info spinStates_buf = spinStates_py.request();
    T * spinStates_ptr = static_cast<T *>(spinStates_buf.ptr);
    for (size_t idx = 0; idx < spinStates_host_.size(); ++idx)
      spinStates_host_[idx] = spinStates_ptr[idx];
    spinStates_dev_ = spinStates_host_;
    nqs_ptr1_->forward(PTR_FROM_THRUST(spinStates_dev_.data()),
                       PTR_FROM_THRUST(lnpsi_dev_.data()), false);
    lnpsi_host_ = lnpsi_dev_;
    return this->make_array_<thrust::complex<T>, std::complex<T>, ComplexToComplex>(lnpsi_host_);
  }

private:
  template <typename T1, typename T2, typename Preprocessor>
  py::array_t<T2> make_array_(const thrust::host_vector<T1> & arr) const
  {
    auto result = py::array_t<T2>(arr.size());
    py::buffer_info result_buf = result.request();
    T2 * result_ptr = static_cast<T2 *>(result_buf.ptr);
    for (size_t idx = 0; idx < arr.size(); ++idx)
      result_ptr[idx] = Preprocessor::transform(arr[idx]);
    return result;
  }

  std::unique_ptr<ansatz<T>> nqs_ptr0_, nqs_ptr1_;
  std::unique_ptr<Sampler4SpinHalf<traits>> sampler_ptr_;
  thrust::host_vector<thrust::complex<T>> spinStates_host_, lnpsi_host_;
  thrust::device_vector<thrust::complex<T>> spinStates_dev_, lnpsi_dev_;
};


PYBIND11_MODULE(_pynqs_gpu, m)
{
  MAKE_PYSAMPLER_MODULE(m, "sRBMSampler",         spinhalf::RBM,          float);
  MAKE_PYSAMPLER_MODULE(m, "dRBMSampler",         spinhalf::RBM,          double);
  MAKE_PYSAMPLER_MODULE(m, "sRBMTrSymmSampler",   spinhalf::RBMTrSymm,    float);
  MAKE_PYSAMPLER_MODULE(m, "dRBMTrSymmSampler",   spinhalf::RBMTrSymm,    double);
  MAKE_PYSAMPLER_MODULE(m, "sRBMZ2PrSymmSampler", spinhalf::RBMZ2PrSymm,  float);
  MAKE_PYSAMPLER_MODULE(m, "dRBMZ2PrSymmSampler", spinhalf::RBMZ2PrSymm,  double);
  MAKE_PYSAMPLER_MODULE(m, "sFFNNSampler",        spinhalf::FFNN,         float);
  MAKE_PYSAMPLER_MODULE(m, "dFFNNSampler",        spinhalf::FFNN,         double);
  MAKE_PYSAMPLER_MODULE(m, "sFFNNTrSymmSampler",  spinhalf::FFNNTrSymm,   float);
  MAKE_PYSAMPLER_MODULE(m, "dFFNNTrSymmSampler",  spinhalf::FFNNTrSymm,   double);
}
