from  dolfin import *
import utils

def recmark(vindex, depth, vec, sub_domain_triangles, visited):
    V = Vertex(mesh, vindex)

    vec[v2d[vindex]] = 1
    eps = 0.1
    if depth > eps and (visited[V.index()] < depth-eps or visited[V.index()] < eps) :
        visited[V.index()] = depth
        for C in V.entities(2):
            sub_domain_triangles[C] = 1
            for v in Cell(mesh, C).entities(0):
                if v != vindex:
                    recmark(v, depth-1, vec,sub_domain_triangles, visited)
    
    if depth < eps:
        for C in V.entities(2):
            sub_domain_triangles[C] = 1
        return None
        
        
mesh=BoundaryMesh(UnitCubeMesh(10,10,10), "exterior")
#mesh.init_cell_orientations(Expression(("x[0]-0.5", "x[1]-0.5", "x[2]-0.5"), degree=1))
#mesh.init()
#mesh, texture = myh5read.readh5("fandisk.h5")
mesh.init()    
#Fandisk
#vertlist = [0,5729]#, 1310] #5729


#Cube
vertlist = [0, 145]

# triangle mesh function for remeshing? 
sub_domain_triangles = MeshFunction("size_t", mesh, 2)
#removedepth = 5
removedepth = 3
visitedverts = MeshFunction("size_t", mesh, 0)
CG = FunctionSpace(mesh, "CG", 1)
v2d = vertex_to_dof_map(CG)


removedverts = Function(CG)
vec = removedverts.vector().get_local()
for start in vertlist:      
    recmark(start, removedepth, vec, sub_domain_triangles, visitedverts)



for c in cells(mesh):        
    counter = 0        
    for e in edges(c):  
        index = e.entities(2)[0] + e.entities(2)[1] - c.index()
        if sub_domain_triangles[index] > 0.5:
            counter += 1 
    if counter >= 2:
        sub_domain_triangles[c.index()] = 1
for e in edges(mesh):
    index0 = e.entities(2)[0]
    index1 = e.entities(2)[1]
    if sub_domain_triangles[index0] > 0.5 and sub_domain_triangles[index1] > 0.5:
        for v in vertices(e):
            vec[v2d[v.index()]] = 1

removedverts.vector().set_local(vec)
File("sub_domain_triangles.pvd") << sub_domain_triangles