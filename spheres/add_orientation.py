import sys
sys.path.append('../')
import dolfin as df
import utils.utils as utils

#mesh = df.Mesh("../meshes/fandisk.xml")
f = utils.ReadObj("../meshes/half-spheres-big-noisy.obj")
mesh = f.function_space().mesh()
mesh.init()

#df.File("fandisk_clean.pvd") << f
utils.Orientate(mesh)
#utils.AddNoise2Mesh(mesh)
utils.SaveMeshH5(mesh,"half-spheres.h5")
df.File("noise.pvd") << f
