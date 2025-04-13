import sys
sys.path.append('../../')
import dolfin as df
import utils.utils
from AugmentedLagrangianScalar import AugmentedLagrangian
from smooth_update import Newton, GradientDescent
from ADMM import ADMM
import numpy as np

mesh = utils.utils.ReadMeshH5("../fandisk.h5") 
mesh.init()

AL = AugmentedLagrangian(mesh, alphas=[0.00001,0.001], rhos=[0.001,0.000001,0.000001], tau = 1e-12)


fout = df.XDMFFile(df.MPI.comm_world, "output/Classic.xdmf")
fout.parameters["flush_output"] = True
fout.parameters["rewrite_function_mesh"] = True

def cb(AL, i):
    AL.normu.assign(df.project( df.sqrt(AL.u**2), df.FunctionSpace(AL.mesh, "DG",0)))
    fout.write(AL.normu, float(i))

ADMM(AL, Newton,cb = cb, max_outer_iter = 300 , max_inner_iter=2, verbose=True)


