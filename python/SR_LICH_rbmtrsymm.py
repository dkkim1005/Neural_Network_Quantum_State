from miscellaneous_tools import argchecker, remove_last_zero_points
from pynqs._pynqs_gpu import SR_float32_RBMTrSymmLICH
from pynqs._pynqs_gpu import SR_float64_RBMTrSymmLICH


def run_gpu(kwargs):
    argchecker(kwargs,
      [    'L',  #  # of lattice sites
          'nf',  #  # of filters
          'ns',  #  # of spin samples for parallel Monte-Carlo
       'nwarm',  #  # of MCMC steps for warming-up
         'nms',  #  # of MCMC steps for sampling spins
         'dev',  #  device number
       'niter',  #  # of iterations to train RBMTrSymm
          'lr',  #  learning_rate
         'rsd',  #  cutoff value of the energy deviation per energy (convergence criterion)
        'seed',  #  seed of the parallel random number generator
        'path',  #  directory to load and save files
       'alpha',  #  exponent in the two-body interaction: J_{i,j} ~ 1/|i-j|^{alpha}
       'theta',  #  J = sin(theta), h = -cos(theta)
         'ver',  #  version
       'dtype',  #  float precision (float32 or float64)
      ])

    # log-file to print a current status
    logfilename = kwargs['path'] + "/RBMTrSymmLICH-L" + str(kwargs['L']) + \
      "NF" + str(kwargs['nf']) + \
      "A" + remove_last_zero_points(str(kwargs['alpha'])) + \
      "T" + remove_last_zero_points(str(kwargs['theta'])) + \
      "V" + str(kwargs['ver']) + \
      "_log.dat"

    with open (logfilename, 'w') as f:
        f.write("#======== PARAMETERS ========\n")
        for item in kwargs.keys():
            key = "{0:>8}".format(str(item))
            value = "{0:>8}".format(str(kwargs[item]))
            f.write("#" + key + " : " + value + "\n")
        f.write("#============================\n")
        
    if kwargs['dtype'] == 'float32':
        SR_float32_RBMTrSymmLICH(kwargs)
    elif kwargs['dtype'] == 'float64':
        SR_float64_RBMTrSymmLICH(kwargs)
    else:
        raise TypeError("The argument \'dtype\' should be either \'float32\' or \'float64\'.")


if __name__ == "__main__":
    kwargs = {
     'L'     : 16,
     'nf'    : 4,
     'ns'    : 1000,
     'nwarm' : 500,
     'nms'   : 1,
     'dev'   : 0, 
     'niter' : 1000,
     'lr'    : 1,
     'rsd'   : 0.001,
     'seed'  : 0,
     'path'  : "./tmp",
     'alpha' : 1,
     'theta' : 1,
     'ver'   : 1,
     'dtype' : 'float64',
   'verbose' : True
     }

    run_gpu(kwargs)
