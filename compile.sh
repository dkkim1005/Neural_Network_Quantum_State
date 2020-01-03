CC=/usr/local/bin/g++
LIBRARIESPATH1=/opt/intel/mkl/lib/intel64
LIBRARIESPATH2=/opt/intel/compilers_and_libraries_2019.1.144/linux/compiler/lib/intel64_lin
TARGET=$1
MAIN=$(echo $TARGET | sed -e 's/.cc$//g')
$CC -O3 -o $MAIN $TARGET -fopenmp -std=c++11 -L$LIBRARIESPATH1 -L$LIBRARIESPATH2 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl $TRNG4
