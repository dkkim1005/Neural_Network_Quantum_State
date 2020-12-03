# Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)
PROJECT (VMC_NEURAL_NET)
CMAKE_MINIMUM_REQUIRED (VERSION 3.0 FATAL_ERROR)

SET (TARGETS "CH-train_rbm;CH-train_fnn;CH-train_rbmtrsymm;CH-train_fnntrsymm")
#SET (USE_INTEL_MKL TRUE)
SET (USE_CUDA TRUE) # --> Currently, only CUDA 10.x version has been tested on this code.
#SET (USE_MAGMA TRUE) # If one use a routine with the MAGMA solver, set it to 'TRUE'.

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

IF (USE_CUDA)
  FIND_PACKAGE (CUDA)
  SET (CUDA_NVCC_FLAGS "-O3 -Xcompiler -fopenmp -std=c++11 -DMIN_CUDA_ARCH=700")
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
  IF (USE_CUDA)
    CUDA_ADD_EXECUTABLE (${TARGET}-gpu ${CMAKE_CURRENT_SOURCE_DIR}/gpu/src/${TARGET}.cu)
  ELSE ()
    ADD_EXECUTABLE (${TARGET}-cpu ${CMAKE_CURRENT_SOURCE_DIR}/cpu/src/${TARGET}.cpp)
  ENDIF ()
ENDFOREACH ()
