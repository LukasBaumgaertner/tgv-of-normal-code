import dolfin as df
import sys
sys.path.append('../utils')
import utils

f = utils.ReadObj("../meshes/noise_block.obj")
mesh = f.function_space().mesh()
utils.Orientate(mesh)
utils.SaveMeshH5(mesh,"block.h5")
df.File("block_noise.pvd") << f
CG = df.VectorFunctionSpace(mesh, "CG", 1)
x = df.project(df.SpatialCoordinate(mesh),CG)

# rescale original mesh to size of noisy one
forig = utils.ReadObj("../meshes/block.obj")
meshorig = forig.function_space().mesh()
CGorig = df.VectorFunctionSpace(meshorig, "CG", 1)
data = df.Function(CGorig)
data.vector().set_local(x.vector())
R = df.FunctionSpace(meshorig, "R", 0)
c = df.Function(R)
ctest = df.TestFunction(R)
ctrial = df.TrialFunction(R)
obj = (c*df.SpatialCoordinate(meshorig) - data)**2*df.dx
dobj = df.derivative(obj, c, ctest)
Hobj = df.derivative(dobj, c, ctrial)
df.solve( Hobj == -dobj, c)
print(float(c), df.assemble(obj))
vec = df.assemble( df.inner(df.TestFunction(CGorig), float(c)*df.SpatialCoordinate(meshorig) - df.SpatialCoordinate(meshorig))*df.dP)
movefun = df.Function(CGorig)
movefun.vector().set_local(vec)
df.ALE.move(meshorig, movefun)
df.File("block_orig.pvd") << movefun

#rTGV
frTGV = utils.ReadObj("block_rTGV.obj")
df.File("block_rTGV.pvd") << frTGV

#meshTGV
fmeshTGV = utils.ReadObj("block_meshTGV.obj")
df.File("block_meshTGV.pvd") << fmeshTGV
