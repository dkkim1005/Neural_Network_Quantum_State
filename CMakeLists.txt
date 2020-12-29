# Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)
PROJECT (VMC_NEURAL_NET)
CMAKE_MINIMUM_REQUIRED (VERSION 3.0 FATAL_ERROR)

SET (USE_INTEL_MKL TRUE)
SET (USE_CUDA_GPU TRUE) # --> Currently, only CUDA 10.x version has been tested on this code.
#SET (USE_MAGMA TRUE) # Should one use a routine with the MAGMA solver, set it to 'TRUE'.

# RBM solver of the 1D quantum Ising model with PBC
SET (TARGETS ${TARGETS} "CH-train_rbmtrsymm")
# FFNN solver of the 1D quantum Ising model with PBC
SET (TARGETS ${TARGETS} "CH-train_ffnntrsymm")
# FFNN solver of the 2D square lattice quantum Ising model with PBC
SET (TARGETS ${TARGETS} "SQ-train_ffnn")
# FFNN solver of the 2D checker-board lattice quantum Ising model
SET (TARGETS ${TARGETS} "CB-train_ffnn")
# RBM solver of the 1D Fermi-Hubbard model
SET (TARGETS ${TARGETS} "fermi_hubbard_CH-train_rbm")
# FFNN solver of the 1D Fermi-Hubbard model
SET (TARGETS ${TARGETS} "fermi_hubbard_CH-train_ffnn")
IF (USE_CUDA_GPU)
  # RBM solver of the 1D long-range quantum Ising model with PBC
  SET (TARGETS ${TARGETS} "LICH-train_rbmtrsymm")
  # RBM solver of the 1D long-range quantum Ising model with OBC
  SET (TARGETS ${TARGETS} "LICH-train_rbm")
  # estimator of Renyi entropy for 'RBMTrSymm' ansatz
  SET (TARGETS ${TARGETS} "meas_renyi_entropy_rbmtrsymm")
  # estimator of Renyi entropy for 'RBM' ansatz
  SET (TARGETS ${TARGETS} "meas_renyi_entropy_rbm")
ENDIF ()

IF (APPLE)
  # Plz check that clang++ is installed at the below directory.
  SET (CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)
ELSEIF (UNIX)
  # One can specify the compiler via 'CMAKE_CXX_COMPILER' variable.
  IF (NOT CMAKE_CXX_COMPILER)
    SET (CMAKE_CXX_COMPILER /usr/bin/g++)
  ENDIF (NOT CMAKE_CXX_COMPILER)
ENDIF ()

IF (USE_INTEL_MKL)
  INCLUDE (${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindMKL.cmake)
  SET (BLAS_LIBRARIES ${MKL_LIBRARIES} ${MKL_OPENMP_LIBRARY})
ELSE ()
  FIND_PACKAGE(BLAS REQUIRED)
ENDIF ()

INCLUDE (${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTRNG4.cmake)

IF (USE_CUDA_GPU)
  FIND_PACKAGE (CUDA)
  SET (CUDA_NVCC_FLAGS "-O3 -Xcompiler -fopenmp -std=c++11 -DMIN_CUDA_ARCH=700 --expt-relaxed-constexpr")
  SET (CUDA_LIBRARIES ${CUDA_LIBRARIES} "cublas")
  IF (USE_MAGMA)
    INCLUDE (${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindMAGMA.cmake)
    SET (CUDA_LIBRARIES ${CUDA_LIBRARIES} ${MAGMA_LIBRARIES} "cusparse" "cudart" "cudadevrt")
  ENDIF ()
ELSE ()
  SET (CMAKE_CXX_FLAGS "-std=c++11 -O3 -fopenmp")
ENDIF ()
LINK_LIBRARIES (${BLAS_LIBRARIES} ${TRNG4_LIBRARIES} ${CUDA_LIBRARIES})
FOREACH (TARGET ${TARGETS})
  IF (USE_CUDA_GPU)
    CUDA_ADD_EXECUTABLE (${TARGET}-gpu ${CMAKE_CURRENT_SOURCE_DIR}/gpu/src/${TARGET}.cu)
  ELSE ()
    ADD_EXECUTABLE (${TARGET}-cpu ${CMAKE_CURRENT_SOURCE_DIR}/cpu/src/${TARGET}.cpp)
  ENDIF ()
ENDFOREACH ()
