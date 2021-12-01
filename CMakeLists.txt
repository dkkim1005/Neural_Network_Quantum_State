# Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10 FATAL_ERROR)

#SET (USE_INTEL_MKL TRUE)
#SET (USE_MAGMA TRUE) # Should one use a routine with the MAGMA solver, set it to 'TRUE'.
#SET (USE_CUDA TRUE)
SET (CPP_STANDARD 14)
SET (OPT_LEVEL 3)

IF (NOT BUILD_PYTHON3_MODULE)
  # RBM solver of the 1D quantum Ising model with PBC
  SET (TARGETS ${TARGETS} CH-train_rbmtrsymm)
  # FFNN solver of the 1D quantum Ising model with PBC
  SET (TARGETS ${TARGETS} CH-train_ffnntrsymm)
  # FFNN solver of the 2D square lattice quantum Ising model with PBC
  SET (TARGETS ${TARGETS} SQ-train_ffnn)
  # FFNN solver of the 2D checker-board lattice quantum Ising model
  SET (TARGETS ${TARGETS} CB-train_ffnn)
  # RBM solver of the 1D Fermi-Hubbard model
  SET (TARGETS ${TARGETS} fermi_hubbard_CH-train_rbm)
  # FFNN solver of the 1D Fermi-Hubbard model
  SET (TARGETS ${TARGETS} fermi_hubbard_CH-train_ffnn)
  IF (USE_CUDA)
    # RBM solver of the 1D long-range quantum Ising model with PBC
    SET (TARGETS ${TARGETS} LICH-train_rbmtrsymm)
    # RBM solver of the 1D long-range quantum Ising model with OBC
    SET (TARGETS ${TARGETS} LICH-train_rbmz2prsymm)
    # estimator of Renyi entropy for 'RBMTrSymm' ansatz
    SET (TARGETS ${TARGETS} meas_renyi_entropy_rbmtrsymm)
    # estimator of Renyi entropy for 'RBMZ2PrSymm' ansatz
    SET (TARGETS ${TARGETS} meas_renyi_entropy_rbmz2prsymm)
  ENDIF ()
ELSE ()
  SET (PYMODULE_NAME _pynqs)
  SET (TARGETS ${TARGETS} pywrapping_sampler)
ENDIF ()

SET (LANGUAGES C CXX)
IF (USE_CUDA)
  SET (LANGUAGES ${LANGUAGES} CUDA)
ENDIF ()
PROJECT (VMC_NEURAL_NET LANGUAGES ${LANGUAGES})

# detecting blas library (CPU-only)
MACRO (SEARCH_BLAS_LIBRARY)
  IF (USE_INTEL_MKL)
    INCLUDE (${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindMKL.cmake)
    IF (MKL_FOUND)
      SET (BLAS_LIBRARIES ${MKL_LIBRARIES} ${MKL_OPENMP_LIBRARY})
    ELSE ()
      MESSAGE(NOTICE "  Other BLAS libraries are being searched...")
      FIND_PACKAGE(BLAS REQUIRED)
    ENDIF ()
  ELSE ()
    FIND_PACKAGE(BLAS REQUIRED)
  ENDIF ()
ENDMACRO ()

# specifying a compiler
#IF (APPLE)
#  # Plz check that clang++ is installed at the below directory.
#  SET (CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)
#ELSEIF (UNIX)
#  # One can specify the compiler via 'CMAKE_CXX_COMPILER' variable.
#  IF (NOT CMAKE_CXX_COMPILER)
#    SET (CMAKE_CXX_COMPILER /usr/bin/g++)
#  ENDIF (NOT CMAKE_CXX_COMPILER)
#ENDIF ()

# compile options
IF (USE_CUDA)
  FIND_PACKAGE (CUDA REQUIRED)
  SET (CUDA_NVCC_FLAGS -O${OPT_LEVEL} -Xcompiler -fopenmp -std=c++${CPP_STANDARD} -DMIN_CUDA_ARCH=700 --expt-relaxed-constexpr)
  LINK_DIRECTORIES (${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  SET (CUDA_LIBRARIES ${CUDA_LIBRARIES} cublas)
  IF (USE_MAGMA)
    SEARCH_BLAS_LIBRARY ()
    INCLUDE (${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindMAGMA.cmake)
    SET (CUDA_LIBRARIES ${CUDA_LIBRARIES} ${MAGMA_LIBRARIES} cusparse cudart cudadevrt)
  ENDIF ()
ELSE ()
  SEARCH_BLAS_LIBRARY ()
  SET (CMAKE_CXX_FLAGS "-std=c++${CPP_STANDARD} -O${OPT_LEVEL}")
  IF (APPLE)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
  ELSEIF (UNIX)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  ENDIF ()
ENDIF ()

# trng4 library
INCLUDE (${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTRNG4.cmake)

# pybind11 library
IF (BUILD_PYTHON3_MODULE)
  IF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
      SET (PYBIND11_GIT_REPOSITORY https://github.com/pybind/pybind11.git)
      MESSAGE (NOTICE " pybind11 library is cloned at the current source directory.")
      EXECUTE_PROCESS (COMMAND git clone ${PYBIND11_GIT_REPOSITORY} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      IF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/CMakeLists.txt)
          MESSAGE (FATAL_ERROR " We can't find CMakeLists.txt! Please check a git repository again.")
      ENDIF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/CMakeLists.txt)
  ENDIF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
  ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
ENDIF ()

LINK_LIBRARIES (${BLAS_LIBRARIES} ${TRNG4_LIBRARIES} ${CUDA_LIBRARIES})

# COMPILE
FOREACH (TARGET ${TARGETS})
  IF (USE_CUDA)
    IF (BUILD_PYTHON3_MODULE)
      MESSAGE (NOTICE ${TARGET})
      CUDA_ADD_LIBRARY("${PYMODULE_NAME}_gpu" MODULE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/src/${TARGET}.cu)
      SET_TARGET_PROPERTIES("${PYMODULE_NAME}_gpu" PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
        CUDA_VISIBILITY_PRESET "hidden"
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
      TARGET_LINK_LIBRARIES("${PYMODULE_NAME}_gpu" pybind11::module)
    ELSE ()
      CUDA_ADD_EXECUTABLE (${TARGET}-gpu ${CMAKE_CURRENT_SOURCE_DIR}/gpu/src/${TARGET}.cu)
    ENDIF ()
  ELSE ()
    IF (BUILD_PYTHON3_MODULE)
      MESSAGE (NOTICE "CPU version of python3 modules are not implemented yet.")
    ELSE ()
      ADD_EXECUTABLE (${TARGET}-cpu ${CMAKE_CURRENT_SOURCE_DIR}/cpu/src/${TARGET}.cpp)
    ENDIF ()
  ENDIF ()
ENDFOREACH ()
