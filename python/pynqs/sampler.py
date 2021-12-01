import numpy as np
from . import _pynqs_gpu


def argchecker(kwargs, ArgCheckList):
    for arg in ArgCheckList:
        if not arg in kwargs:
            raise Exception ('You omit an essential argument registered in :', ArgCheckList)


class RBM:
    def __init__(self, **kwargs):
        """
          floatType : 'float32' or 'float64'
          symmType  : 'None' --> No symmetry is considerd.
                      'tr' --> Translational symmetry is considered.
                      'z2pr' --> Z2 and parity symmetries are considered.
        """
        argchecker(kwargs, ['floatType', 'symmType'])

        floatType = kwargs['floatType']
        self._floatType = floatType
        symmType = kwargs['symmType']
        self._symmType = symmType

        if floatType == 'float32' and symmType == 'None':
            self._sampler = _pynqs_gpu.sRBMSampler
        elif floatType == 'float64' and symmType == 'None':
            self._sampler = _pynqs_gpu.dRBMSampler
        elif floatType == 'float32' and symmType == 'tr':
            self._sampler = _pynqs_gpu.sRBMTrSymmSampler
        elif floatType == 'float64' and symmType == 'tr':
            self._sampler = _pynqs_gpu.dRBMTrSymmSampler
        elif floatType == 'float32' and symmType == 'z2pr':
            self._sampler = _pynqs_gpu.sRBMZ2PrSymmSampler
        elif floatType == 'float64' and symmType == 'z2pr':
            self._sampler = _pynqs_gpu.dRBMZ2PrSymmSampler
        else:
            raise Exception(' --hint:  floatType: float32 or float64 / symmType: None, tr, z2pr')


    def init(self, **kwargs):
        argchecker(kwargs, ['nInputs', 'nHiddens', 'nChains', 'seedNumber',
                            'seedDistance', 'path_to_load', 'init_mcmc_steps'])

        self._rbm = self._sampler(kwargs)
        self._nInputs = int(kwargs['nInputs'])
        self._nChains = int(kwargs['nChains'])
        path = str(kwargs['path_to_load'])
        init_mcmc_steps = int(kwargs['init_mcmc_steps'])
        self._rbm.load('%s'%path)
        self._rbm.warm_up(init_mcmc_steps)


    def do_mcmc_steps(self, mcmc_steps):
        self._rbm.do_mcmc_steps(mcmc_steps)


    def get_spinStates(self):
        return self._rbm.get_spinStates().reshape([-1, self._nInputs])


    def get_lnpsi(self):
        return self._rbm.get_lnpsi()


    def get_lnpsi_for_fixed_spins(self, spinStates):
        spinStates = np.array(spinStates). \
                      astype(self._floatType). \
                      reshape([self._nChains, self._nInputs])
        return self._rbm.get_lnpsi_for_fixed_spins(spinStates)
