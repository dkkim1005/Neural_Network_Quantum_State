# Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)
#!/usr/bin/env bash
#====================================================
SRCS=(SQ-train_fnn.cu CB-train_fnn.cu) # source list to compile the cuda executable files
MIN_CUDA_ARCH=700 # minimum cuda architecture
CFLAGS="-O3 -std=c++11"
CC=DEFAULT # CUDA nvcc compiler
TRNG4_INC_PATH=DEFAULT
TRNG4_LIB_PATH=DEFAULT
MKL_INC_PATH=DEFAULT
MKL_LIB_PATH1=DEFAULT
MKL_LIB_PATH2=DEFAULT
MAGMA_INC_PATH=DEFAULT
MAGMA_LIB_PATH=DEFAULT
CUDA_INC_PATH=DEFAULT
CUDA_LIB_PATH=DEFAULT
#====================================================

if [ $CC = "DEFAULT" ]; then
  CC=$(which nvcc)
  if [ -z $CC ]; then
    echo "nvcc compiler is not detected. Plz specify the path of nvcc."
    exit 1
  fi
fi

#---------default path---------
# trng4
if [ $TRNG4_INC_PATH = "DEFAULT" ]; then
  TRNG4_INC_PATH=/usr/local/include/trng
fi
if [ $TRNG4_LIB_PATH = "DEFAULT" ]; then
  TRNG4_LIB_PATH=/usr/local/lib
fi
# MKL
if [ $MKL_INC_PATH = "DEFAULT" ]; then
  MKL_INC_PATH=/opt/intel/mkl/include
fi
if [ $MKL_LIB_PATH1 = "DEFAULT" ]; then
  MKL_LIB_PATH1=/opt/intel/mkl/lib/intel64
fi
if [ $MKL_LIB_PATH2 = "DEFAULT" ]; then
  MKL_LIB_PATH2=/opt/intel/compilers_and_libraries/linux/lib/intel64
fi
# MAGMA-GPU
if [ $MAGMA_INC_PATH = "DEFAULT" ]; then
  MAGMA_INC_PATH=/usr/local/magma/include
fi
if [ $MAGMA_LIB_PATH = "DEFAULT" ]; then
  MAGMA_LIB_PATH=/usr/local/magma/lib
fi
# CUDA
if [ $CUDA_INC_PATH = "DEFAULT" ]; then
  CUDA_INC_PATH=/usr/local/cuda/include
fi
if [ $CUDA_LIB_PATH = "DEFAULT" ]; then
  CUDA_LIB_PATH=/usr/local/cuda/lib64
fi
#------------------------------

TRNG4_FLAGS="-ltrng4 -I$TRNG4_INC_PATH -L$TRNG4_LIB_PATH"
MKL_FLAGS="-I$MKL_INC_PATH -L$MKL_LIB_PATH1 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L$MKL_LIB_PATH2 -liomp5 -lpthread -lm -ldl"
MAGMA_FLAGS="-Xcompiler -fopenmp -DMIN_CUDA_ARCH=$MIN_CUDA_ARCH -I$MAGMA_INC_PATH -I$CUDA_INC_PATH -L$MAGMA_LIB_PATH -L$CUDA_LIB_PATH -lmagma_sparse -lmagma -lcublas -lcusparse -lcudart -lcudadevrt"

for SRC in ${SRCS[@]}; do
  TARGET=$(echo $SRC | sed -e 's/.cu$//g')
  echo -e "\e[1;7;32mCompiling... ($SRC => $TARGET)\e[0m"
  CMD="$CC -o $TARGET $SRC $CFLAGS $MAGMA_FLAGS $MKL_FLAGS $TRNG4_FLAGS"
  echo $CMD
  eval $CMD
done
