import sys
sys.path.append('../')
import dolfin as df
import utils.utils as utils

#mesh = df.Mesh("../meshes/fandisk.xml")
f = utils.ReadObj("../meshes/noise_fandisk.obj")
mesh = f.function_space().mesh()

#df.File("fandisk_clean.pvd") << f
utils.Orientate(mesh)
#utils.AddNoise2Mesh(mesh)
utils.SaveMeshH5(mesh,"fandisk.h5")
df.File("noise.pvd") << f

f1 = utils.ReadObj("./meshTGV.obj")
df.File("meshTGV.pvd") << f1
f2 = utils.ReadObj("./rTGV.obj")
df.File("rTGV.pvd") << f2
f3 = utils.ReadObj("../meshes/fandisk.obj")
df.File("orig.pvd") << f3