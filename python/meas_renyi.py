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
  'seedDistance' : 123456789,
  'init_mcmc_steps' : 300
}
# transverse-field strengthes
hfield = '-1'
# functor to locate a path of the file
filepath = './temp/build/RBMTrSymmCH-N%dA%dH%sV1'\
            %(kwargs['nInputs'], kwargs['nHiddens'], hfield)
kwargs['path_to_load'] = filepath
# total number of measurements
nmeas = 1000
# number of Monte-Carlo steps
nms = 20
# length of the subregion 
l = kwargs['nInputs']//2
# range of the error bar (95% confidence)
Z = 2

def swap_operations(spins0, spins1, l):
    spins2 = spins0.copy()
    spins3 = spins1.copy()
    spins2[:,:l] = spins1[:,:l].copy()
    spins3[:,:l] = spins0[:,:l].copy()
    return spins2, spins3

# select rbm architectures
rbms = [sampler.RBM(floatType = floatType, symmType = symmType), \
        sampler.RBM(floatType = floatType, symmType = symmType)]
for i, rbm in enumerate(rbms):
    kwargs['seedNumber'] = (i+1)*kwargs['seedDistance']
    rbm.init(**kwargs)

tr2 = np.zeros([nmeas], dtype = floatType)
for i in range(nmeas):
    print ('# of measurements: %d'%i, end = '\r')
    rbms[0].do_mcmc_steps(nms)
    rbms[1].do_mcmc_steps(nms)
    spins0 = rbms[0].get_spinStates()
    spins1 = rbms[1].get_spinStates()
    lnpsi_0 = rbms[0].get_lnpsi()
    lnpsi_1 = rbms[1].get_lnpsi()
    spins2, spins3 = swap_operations(spins0, spins1, l)
    lnpsi_2 = rbms[0].get_lnpsi_for_fixed_spins(spins2)
    lnpsi_3 = rbms[1].get_lnpsi_for_fixed_spins(spins3)
    tr2[i] = np.mean(np.exp(lnpsi_2+lnpsi_3-lnpsi_0-lnpsi_1)).real
renyi = -np.log(np.mean(tr2))
renyi_err = Z*np.sqrt(np.sum((tr2-np.mean(tr2))**2)/(nmeas*(nmeas-1)))
print ('R\'enyi: %.5E'%(renyi), ' +/- %.3E'%renyi_err)
