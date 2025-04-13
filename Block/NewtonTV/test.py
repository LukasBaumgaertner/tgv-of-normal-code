import sys
sys.path.append('../../')
import dolfin as df
import utils.utils
from AugmentedLagrangianScalar import AugmentedLagrangian
from smooth_update import Newton, GradientDescent
from ADMM import ADMM
import numpy as np

mesh = utils.utils.ReadMeshH5("../block.h5") 
mesh.init()

#AL = AugmentedLagrangian(mesh, alphas=[0.1,0.2], rhos=[10,10,10], tau = 1e-4)
#AL = AugmentedLagrangian(mesh, alphas=[0.000001,0.005], rhos=[0.01,0.001,0.001], tau = 1e-10)
#AL = AugmentedLagrangian(mesh, alphas=[0.0001,0.004], rhos=[0.001,0.00001,0.00001], tau = 1e-14)
#AL = AugmentedLagrangian(mesh, alphas=[0.0001,0.003], rhos=[0.001,0.00001,0.00001], tau = 1e-14)
#AL = AugmentedLagrangian(mesh, alphas=[0.00006,0.004], rhos=[0.001,0.00001,0.00001], tau = 1e-14)
#AL = AugmentedLagrangian(mesh, alphas=[0.00008,0.003], rhos=[0.001,0.00001,0.00001], tau = 1e-14)
AL = AugmentedLagrangian(mesh, alphas=[0.00005,0.003], rhos=[0.001,0.00001,0.00001], tau = 1e-12)
"""
AL.update_nonsmooth()
AL.update_aux()
AL.update_multipliers()

import taylortest
def Hess_action(d):
    dfun0 = df.Function(AL.CG)
    dfun0.vector().set_local(d[0])
    dfun1 = df.Function(AL.RTdof)
    dfun1.vector().set_local(d[1])
    return AL.Hess(d=[dfun0.vector(), dfun1.vector()])

gradf = lambda x: [AL.derivative(x)[0].get_local(), 0*x[1] ]
print(np.linalg.norm(AL.udof.vector().get_local()))
taylortest.run_secondorder([AL.tracking_data.vector().get_local(), AL.udof.vector().get_local()], AL.val, gradf, Hess_action,d = [AL.tracking_data.vector().get_local(), 0*AL.udof.vector().get_local()],  maxcounters = 30, h=1e-2, hfactor = 1.2)
exit()
"""
fout = df.XDMFFile(df.MPI.comm_world, "output/Classic.xdmf")
fout.parameters["flush_output"] = True
fout.parameters["rewrite_function_mesh"] = True

def cb(AL, i):
    AL.normu.assign(df.project( df.sqrt(AL.u**2), df.FunctionSpace(AL.mesh, "DG",0)))
    fout.write(AL.normu, float(i))

ADMM(AL, Newton,cb = cb, max_outer_iter = 200 , max_inner_iter=3, verbose=True, with_aux=False)

