#define __KISTI_GPU__

#include <memory>
#include "../include/neural_quantum_state.cuh"
#include "../include/common.cuh"
#include "../include/hamiltonians.cuh"
#include "../include/optimizer.cuh"
#include "../include/meas.cuh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

/*
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
*/

#ifdef __KISTI_GPU__
// =====================================================================
// The below codes are written for running KISTI GPU machines
//

// Part 1: STOCHASTIC RECONFIGURATION (RBMTrSymmLICH)
template <typename FloatType>
std::string remove_zeros_in_str(const FloatType val);

using namespace spinhalf;

template <typename T>
void optimize_RBMTrSymmLICH(const py::dict kwargs)
{
  const int L = kwargs["L"].cast<int>(),
    nChains = kwargs["ns"].cast<int>(),
    nWarmup = kwargs["nwarm"].cast<int>(),
    nMonteCarloSteps = kwargs["nms"].cast<int>(),
    deviceNumber = kwargs["dev"].cast<int>(),
    nIterations =  kwargs["niter"].cast<int>();

  const T lr = kwargs["lr"].cast<T>(),
    RSDcutoff = kwargs["rsd"].cast<T>();

  const unsigned long long seed = kwargs["seed"].cast<unsigned long long>();

  const std::string path = kwargs["path"].cast<std::string>() + "/",
    Lstr = std::to_string(L);

  const int nf = kwargs["nf"].cast<int>();
  const T alpha = kwargs["alpha"].cast<T>();
  const int ver = kwargs["ver"].cast<int>();
  const T theta = kwargs["theta"].cast<T>();

  // check whether the cuda device is available
  int devicesCount;
  CHECK_ERROR(cudaSuccess, cudaGetDeviceCount(&devicesCount));
  if (deviceNumber >= devicesCount)
  {
    std::cerr << "# error ---> dev(" << deviceNumber << ") >= # of devices(" << devicesCount << ")" << std::endl;
    exit(1);
  }
  CHECK_ERROR(cudaSuccess, cudaSetDevice(deviceNumber));

  struct SamplerTraits { using AnsatzType = RBMTrSymm<T>; using FloatType = T; };

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(nIterations)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(L)*
    static_cast<unsigned long>(nChains);

  const std::string verstr = std::to_string(ver),
    nfstr = std::to_string(nf),
    alphastr = remove_zeros_in_str(alpha),
    thetastr = remove_zeros_in_str(theta);
  RBMTrSymm<T> machine(L, nf, nChains);
  const T J = std::sin(theta), h = -std::cos(theta);
  // load parameters
  const std::string prefix = path + "RBMTrSymmLICH-L" + Lstr + "NF" + nfstr + "A" + alphastr + "T" + thetastr + "V" + verstr,
    logpath = prefix + "_log.dat";
  machine.load(prefix);
  // Transverse Field Ising Hamiltonian with long-range interaction on the 1D chain lattice
  LITFIChain<SamplerTraits> sampler(machine, L, h, J, alpha, true, seed, nBlocks, prefix);
  sampler.warm_up(nWarmup);
  StochasticReconfigurationCG<T> iTimePropagator(nChains, machine.get_nVariables());
  iTimePropagator.propagate(sampler, nIterations, nMonteCarloSteps, lr, RSDcutoff, logpath);
  // save parameters
  machine.save(prefix);
}

template <typename FloatType>
std::string remove_zeros_in_str(const FloatType val)
{
  std::string tmp = std::to_string(val);
  tmp.erase(tmp.find_last_not_of('0') + 1, std::string::npos);
  tmp.erase(tmp.find_last_not_of('.') + 1, std::string::npos);
  return tmp;
}


// Part 2: X-X CORRELATION (RBMTrSymmLICH)
template <typename T>
void xx_correlation_RBMTrSymmLICH(const py::dict kwargs)
{
  const int L = kwargs["L"].cast<int>(),
    nf = kwargs["nf"].cast<int>(),
    nChains = kwargs["ns"].cast<int>(),
    niter = kwargs["niter"].cast<int>(),
    nWarmup = kwargs["nwarm"].cast<int>(),
    nMonteCarloSteps = kwargs["nms"].cast<int>(),
    deviceNumber = kwargs["dev"].cast<int>();

  const unsigned long seed = kwargs["seed"].cast<unsigned long>();

  // check whether the cuda device is available
  int devicesCount;
  CHECK_ERROR(cudaSuccess, cudaGetDeviceCount(&devicesCount));
  if (deviceNumber >= devicesCount)
  {
    std::cerr << "# error ---> dev(" << deviceNumber << ") >= # of devices(" << devicesCount << ")" << std::endl;
    exit(1);
  }
  CHECK_ERROR(cudaSuccess, cudaSetDevice(deviceNumber));

  RBMTrSymm<T> psi(L, nf, nChains);

  const std::string filepath = kwargs["path"].cast<std::string>() + "/" + kwargs["filename"].cast<std::string>();

  // load parameters: w,a,b
  psi.load(filepath);

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(niter)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(L)*
    static_cast<unsigned long>(nChains);

  // measurements of the overlap integral for the given wave functions
  struct TRAITS { using AnsatzType = RBMTrSymm<T>; using FloatType = T; };

  Sampler4SpinHalf<TRAITS> smp(psi, seed, nBlocks);
  MeasSpinXSpinXCorrelation<TRAITS> corr(smp, psi);
  std::vector<T> ss(L*L, 0), s(L, 0);
  const std::string logfile = kwargs["path"].cast<std::string>() +
    "/X-" + kwargs["filename"].cast<std::string>() + "_log.dat";
  corr.measure(niter, nMonteCarloSteps, nWarmup, ss.data(), s.data(), logfile);

  const std::string filename1 = kwargs["path"].cast<std::string>() +
    "/Corr-XX-" + kwargs["filename"].cast<std::string>() + "-TAG" + std::to_string(kwargs["tag"].cast<int>()) + ".dat",
    filename2 = kwargs["path"].cast<std::string>() +
      "/Mag-X-" + kwargs["filename"].cast<std::string>() + "-TAG" + std::to_string(kwargs["tag"].cast<int>()) + ".dat";

  std::ofstream wfile1(filename1), wfile2(filename2);
  for (int i=0; i<L; ++i)
  {
    wfile2 << s[i] << " ";
    for (int j=0; j<L; ++j)
      wfile1 << ss[i*L+j] << " ";
    wfile1 << std::endl;
  }
  wfile1.close();
  wfile2.close();
}


// Part 3: Z-Z CORRELATION (RBMTrSymmLICH)
template <typename T>
void zz_correlation_RBMTrSymmLICH(const py::dict kwargs)
{
  const int L = kwargs["L"].cast<int>(),
    nf = kwargs["nf"].cast<int>(),
    nChains = kwargs["ns"].cast<int>(),
    niter = kwargs["niter"].cast<int>(),
    nWarmup = kwargs["nwarm"].cast<int>(),
    nMonteCarloSteps = kwargs["nms"].cast<int>(),
    deviceNumber = kwargs["dev"].cast<int>();

  const unsigned long seed = kwargs["seed"].cast<unsigned long>();

  // check whether the cuda device is available
  int devicesCount;
  CHECK_ERROR(cudaSuccess, cudaGetDeviceCount(&devicesCount));
  if (deviceNumber >= devicesCount)
  {
    std::cerr << "# error ---> dev(" << deviceNumber << ") >= # of devices(" << devicesCount << ")" << std::endl;
    exit(1);
  }
  CHECK_ERROR(cudaSuccess, cudaSetDevice(deviceNumber));

  RBMTrSymm<T> psi(L, nf, nChains);

  const std::string filepath = kwargs["path"].cast<std::string>() + "/" + kwargs["filename"].cast<std::string>();

  // load parameters: w,a,b
  psi.load(filepath);

  // block size for the block splitting scheme of parallel Monte-Carlo
  const unsigned long nBlocks = static_cast<unsigned long>(niter)*
    static_cast<unsigned long>(nMonteCarloSteps)*
    static_cast<unsigned long>(L)*
    static_cast<unsigned long>(nChains);

  // measurements of the overlap integral for the given wave functions
  struct TRAITS { using AnsatzType = RBMTrSymm<T>; using FloatType = T; };

  Sampler4SpinHalf<TRAITS> smp(psi, seed, nBlocks);
  MeasSpinZSpinZCorrelation<TRAITS> corr(smp);
  std::vector<T> ss(L*L, 0);
  const std::string logfile = kwargs["path"].cast<std::string>() +
    "/Z-" + kwargs["filename"].cast<std::string>() + "_log.dat";
  corr.measure(niter, nMonteCarloSteps, nWarmup, ss.data(), logfile);

  const std::string filename = kwargs["path"].cast<std::string>() +
    "/Corr-" + kwargs["filename"].cast<std::string>() + "-TAG" + std::to_string(kwargs["tag"].cast<int>()) + ".dat";
  std::ofstream wfile(filename);
  for (int i=0; i<L; ++i)
  {
    for (int j=0; j<L; ++j)
      wfile << ss[i*L+j] << " ";
    wfile << std::endl;
  }
  wfile.close();
}

// =====================================================================
#endif // __KISTI_GPU__



PYBIND11_MODULE(_pynqs_gpu, m)
{
  /*
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
  */

#ifdef __KISTI_GPU__
  m.def("SR_float32_RBMTrSymmLICH", &optimize_RBMTrSymmLICH<float>, "");
  m.def("SR_float64_RBMTrSymmLICH", &optimize_RBMTrSymmLICH<double>, "");
  m.def("xx_corr_float32_RBMTrSymmLICH", &xx_correlation_RBMTrSymmLICH<float>, "");
  m.def("xx_corr_float64_RBMTrSymmLICH", &xx_correlation_RBMTrSymmLICH<double>, "");
  m.def("zz_corr_float32_RBMTrSymmLICH", &zz_correlation_RBMTrSymmLICH<float>, "");
  m.def("zz_corr_float64_RBMTrSymmLICH", &zz_correlation_RBMTrSymmLICH<double>, "");
#endif
}
