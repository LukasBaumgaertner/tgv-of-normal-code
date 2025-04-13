import dolfin as df
import utils

mesh = df.Mesh("../fandisk/fandisk.xml")
utils.Orientate(mesh)
utils.AddNoise2Mesh(mesh)
utils.SaveMeshH5(mesh,"fandisk/fandisk.h5")


