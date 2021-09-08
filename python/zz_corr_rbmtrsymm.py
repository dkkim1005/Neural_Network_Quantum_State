from miscellaneous_tools import argchecker, remove_last_zero_points
from pynqs._pynqs_gpu import zz_corr_float32_RBMTrSymmLICH
from pynqs._pynqs_gpu import zz_corr_float64_RBMTrSymmLICH


def run_gpu(kwargs):
    argchecker(kwargs,
      [    'L',  #  # of lattice sites
          'nf',  #  # of filters
          'ns',  #  # of spin samples for parallel Monte-Carlo
       'niter',  #  # of iterations to train RBMTrSymm
       'nwarm',  #  # of MCMC steps for warming-up
         'nms',  #  # of MCMC steps for sampling spins
         'dev',  #  device number
        'seed',  #  seed of the parallel random number generator
        'path',  #  directory to load and save files
    'filename',  #  data file of the trained model
         'tag',  #  tag of results
       'dtype',  #  float precision (float32 or float64)
      ])

    # log-file to print a current status
    logfilename = kwargs['path'] + "/Z-" + kwargs['filename'] + "_log.dat"

    with open (logfilename, 'w') as f:
        f.write("#======== PARAMETERS ========\n")
        for item in kwargs.keys():
            key = "{0:>8}".format(str(item))
            value = "{0:>8}".format(str(kwargs[item]))
            f.write("#" + key + " : " + value + "\n")
        f.write("#============================\n")

    if kwargs['dtype'] == 'float32':
        zz_corr_float32_RBMTrSymmLICH(kwargs)
    elif kwargs['dtype'] == 'float64':
        zz_corr_float64_RBMTrSymmLICH(kwargs)
    else:
        raise TypeError("The argument \'dtype\' should be either \'float32\' or \'float64\'.")
