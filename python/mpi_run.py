#!/usr/bin/env python3
import os
from mpi4py import MPI
from miscellaneous_tools import remove_last_zero_points
import SR_LICH_rbmtrsymm
import xx_corr_rbmtrsymm
import zz_corr_rbmtrsymm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dev = rank
L = 128
nf = 4
alpha = 1.0
ns_1 = 3000
ns_2 = 4000
niter_1 = 80000
niter_2 = 1000
nms = 5
nwarm = 500
ver = 1
lr = 1.0
rsd = 0.001
seed = 1234567
ntag = 5
thetas = [[1.02, 1.04], # rank 0
          [1.06, 1.08]  # rank 1
         ]

sr_parameters = {
  'dev'   : dev,
  'L'     : L,
  'nf'    : nf,
  'alpha' : alpha,
  'ns'    : ns_1,
  'niter' : niter_1,
  'nms'   : nms,
  'nwarm' : nwarm,
  'ver'   : ver ,
  'lr'    : lr,
  'rsd'   : rsd,
  'seed'  : seed,
  'dtype' : 'float32'
}

if comm.size != 2:
    print ('# CHECK THE NUBER OF PROCESSES YOU SUBMITTED.')
    exit (1)

path = f'../LITFI/pbc/A{alpha}/L{L}/'
os.makedirs(path, exist_ok = True)
sr_parameters['path'] = path
str_alpha = remove_last_zero_points(str(alpha))

for theta in thetas[rank]:
    sr_parameters['theta'] = theta
    SR_LICH_rbmtrsymm.run_gpu(sr_parameters)
    str_theta = remove_last_zero_points(str(theta))
    filename = f"RBMTrSymmLICH-L{L}NF{nf}A{str_alpha}T{str_theta}V{ver}"
    seed0 = seed
    for tag in range(ntag):
        seed0 += 1234567
        meas_parameters = {
            'dev'      : dev,
            'L'        : L,
            'nf'       : nf,
            'ns'       : ns_2,
            'niter'    : niter_2,
            'nms'      : nms,
            'nwarm'    : nwarm,
            'path'     : path,
            'seed'     : seed0,
            'tag'      : tag,
            'filename' : filename,
            'dtype'    : 'float32'
        }
        xx_corr_rbmtrsymm.run_gpu(meas_parameters)
        zz_corr_rbmtrsymm.run_gpu(meas_parameters)
