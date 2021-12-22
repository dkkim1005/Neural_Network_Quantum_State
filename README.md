# Neural Network Quantum State
Neural network ansatz to approximate a ground state by using variational Monte Carlo (VMC)

## Required compilers
+ c++11 or higher
+ CUDA 10.x.x or higher


Build reciepe 1 [C++ or CUDA]
--------------
    mkdir build
    cd build
    cmake ../ -DUSE_CUDA=TRUE # <- use this flag to run a gpu device.


Build reciepe 2 [Python-CUDA]
--------------
    # One can import 'pynqs' module. The examples are listed in './python' folder.
    python3 ./setup.py build
    export PYTHONPATH=$(pwd)/python:$PYTHONPATH

Bug report
--------------
When you encounter bugs or problems by using this code, please let us know through the email address as following. <br />
dkkim1005@gmail.com



Reference
--------------
1) G. Carleo and M. Troyer, *Solving the quantum many-body problem with artificial neural networks*, Science **355**, 602 (2017).
   [arXiv link](https://arxiv.org/abs/1606.02318?context=cond-mat)
