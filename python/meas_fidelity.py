#!/usr/bin/env python3
import numpy as np
from pynqs import sampler

floatType = 'float32'
symmType = 'tr'
# hyper parameter sets of rbm and MCMC sampler
kwargs = {
  'nInputs' : 16,
  'nHiddens' : 4,
  'nChains' : 1000,
  'seedNumber' : 0,
  'seedDistance' : 123456789,
  'path_to_load' : None,
  'init_mcmc_steps' : 300
}
# transverse-field strengthes
hfields = ['-0.9', '-1.1']
# functor to locate a path of the file
filepath = lambda hfield : './temp/build/RBMTrSymmCH-N%dA%dH%sV1'\
                  %(kwargs['nInputs'], kwargs['nHiddens'], str(hfield))
# total number of measurements
nmeas = 100
# number of Monte-Carlo steps
nms = 20
# range of the error bar (95% confidence)
Z = 2

# select rbm architectures
rbms = [sampler.RBM(floatType = floatType, symmType = symmType), \
        sampler.RBM(floatType = floatType, symmType = symmType)]

for i, hfield in enumerate(hfields):
    kwargs['path_to_load'] = filepath(hfield)
    rbms[i].init(**kwargs)

F2 = np.zeros([nmeas], dtype = floatType)
for i in range(nmeas):
    print ('# of measurements: %d'%i, end = '\r')
    rbms[0].do_mcmc_steps(nms)
    rbms[1].do_mcmc_steps(nms)
    spins0 = rbms[0].get_spinStates()
    spins1 = rbms[1].get_spinStates()
    lnpsi_00 = rbms[0].get_lnpsi()
    lnpsi_11 = rbms[1].get_lnpsi()
    lnpsi_01 = rbms[0].get_lnpsi_for_fixed_spins(spins1)
    lnpsi_10 = rbms[1].get_lnpsi_for_fixed_spins(spins0)
    F2[i] = np.mean(np.exp(lnpsi_01 - lnpsi_00)*np.exp(lnpsi_10 - lnpsi_11)).real

F_mean = np.sqrt(np.mean(F2))
F_err = Z*np.sqrt(np.sum((F2 - np.mean(F2))**2)/(nmeas*(nmeas-1)))/2.0
print ('fidelity : %.5E'%F_mean, ' +/- %.3E'%F_err)
