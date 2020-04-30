# Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)
#!/usr/bin/env bash
#====================================================
SRCS=(SQ-train_fnn.cu CB-train_fnn.cu) # source list to compile the cuda executable files
MIN_CUDA_ARCH=700 # minimum cuda architecture
CFLAGS="-O3 -std=c++11"
CC=DEFAULT # CUDA nvcc compiler
TRNG4_INC_PATH=DEFAULT
TRNG4_LIB_PATH=DEFAULT
MAGMA_PKG_CONFIG_PATH=DEFAULT
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
# MAGMA-GPU
if [ $MAGMA_PKG_CONFIG_PATH = "DEFAULT" ]; then
  MAGMA_PKG_CONFIG_PATH=/usr/local/magma/lib/pkgconfig
fi
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
MAGMA_FLAGS="-DMIN_CUDA_ARCH=$MIN_CUDA_ARCH -I$MAGMA_INC_PATH -I$CUDA_INC_PATH -L$MAGMA_LIB_PATH -L$CUDA_LIB_PATH `pkg-config $MAGMA_PKG_CONFIG_PATH/magma.pc --libs | sed 's/-fopenmp/-Xcompiler -fopenmp/g'`"

# name of the bash script
SCRIPT_NAME=$(basename "$0")
# location of the bash script
PREFIX=$(echo $BASH_SOURCE | sed -e "s/$SCRIPT_NAME$//g")

for SRC in ${SRCS[@]}; do
  TARGET=$(echo $SRC | sed -e 's/.cu$//g')
  SRC_PATH=$(find $PREFIX -name $SRC)
  if [ -z $SRC_PATH ]; then
    echo -e "\e[41m${SRC} is not exist...\e[0m"
    continue
  fi
  echo -e "\e[1;7;32mCompiling... ($SRC => $TARGET)\e[0m"
  CMD="$CC -o $TARGET $SRC_PATH $CFLAGS $MAGMA_FLAGS $TRNG4_FLAGS"
  echo $CMD
  eval $CMD
done
