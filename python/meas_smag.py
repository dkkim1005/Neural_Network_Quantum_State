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
  'init_mcmc_steps' : 300
}
# transverse-field strengthes
hfield = '-1.1'
# functor to locate a path of the file
filepath = './temp/build/RBMTrSymmCH-N%dA%dH%sV1'\
            %(kwargs['nInputs'], kwargs['nHiddens'], hfield)
kwargs['path_to_load'] = filepath
# total number of measurements
nmeas = 1000
# number of Monte-Carlo steps
nms = 20
# range of the error bar (95% confidence)
Z = 2

rbm = sampler.RBM(floatType = floatType, symmType = symmType)
rbm.init(**kwargs)

mag = np.zeros([nmeas], dtype = floatType)
for i in range(nmeas):
    print ('# of measurements: %d'%i, end = '\r')
    rbm.do_mcmc_steps(nms)
    spinStates = rbm.get_spinStates()
    mag[i] = np.mean(np.abs(np.mean(spinStates, axis = 1)))
mag_mean = np.mean(mag)
mag_err = Z*np.sqrt(np.sum((mag - mag_mean)**2)/(nmeas*(nmeas-1)))
print ('<|m|> : %.5E'%mag_mean, ' +/- %.3E'%mag_err)
