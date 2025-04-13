import numpy as np
import dolfin as df 
import os

def AddNoise2Mesh(mesh, seed = 0, Sigma = 0.2, move = np.array([1.0]),rescale_sigma_to_edgelength=True):
    if rescale_sigma_to_edgelength:
        HDT = df.FunctionSpace(mesh, "HDiv Trace", 0)
        dofmapHDT = HDT.dofmap()
        hdtest= df.TestFunction(HDT)
        l = df.assemble( hdtest('+')*df.Constant(1.0)*df.dS(domain=mesh) + hdtest('+')*df.Constant(1.0)*df.ds(domain=mesh)).get_local() #edgelengths
    np.random.seed(seed) 
    dofs = df.FunctionSpace(mesh,'DG',0).dofmap()
    V = df.vertex_to_dof_map(df.FunctionSpace(mesh,'CG',1))
    space = df.VectorFunctionSpace(mesh,'CG', 1, dim=3)
    noise = df.Function(space) 
    loc = noise.vector().local_size()
    normal = df.project(df.CellNormal(mesh), df.VectorFunctionSpace(mesh, "DG", 0))
    cvol = df.project(df.CellVolume(mesh), df.FunctionSpace(mesh, "DG", 0))
    values = np.zeros(loc)
    
    
    
    for v in df.vertices(mesh):
        w = np.zeros(3)
        i = V[v.index()]
        if rescale_sigma_to_edgelength:
            edgecounter = 0
            lensum = 0
            for e in df.edges(v):
                edof = dofmapHDT.entity_dofs(mesh, 1, [e.index()])
                lensum += l[edof[0]]
                edgecounter += 1
            lenavg = lensum/edgecounter
            Sigma_vertex = lenavg*Sigma
        else:
            Sigma_vertex = Sigma
        for c in df.cells(v):
            k = c.index()
            dof = dofs.cell_dofs(k)[0]
            w += np.array([(normal.vector()[3*dof+0],normal.vector()[3*dof+1],normal.vector()[3*dof+2])]).reshape(3)/cvol.vector()[dof]
        w = w/np.linalg.norm(w)
        values[(3*i+0):(3*i+3)] = move[i % move.size]*np.random.normal(scale = Sigma_vertex)*w
    noise.vector().set_local(values)
    noise.vector().apply("")
    df.ALE.move(mesh, noise)
    mesh.init()
    return mesh
  
def IsSame(c1, c2):
    ''' Checks wether the two cells have the same orientation

    '''
    c1V = np.zeros(3, dtype=int)
    c2V = np.zeros(3, dtype=int)
    i=0
    for v in df.vertices(c1):
        c1V[i] = v.index()
        i=i+1
    i=0
    for v in df.vertices(c2):
        c2V[i] = v.index()
        i = i+1

    combos = np.zeros([2,3])
    counter = 0 # counts common vertices - at least 1 common vertex when calling from Orientate
    for i in range(3): 
        for j in range(3):
            if  c1V[i] == c2V[j]:
                combos[0, counter] = i
                combos[1, counter] = j
                counter += 1
    if counter==3:
        print("c1 = ", c1.index())
        print("c2 = ", c2.index())
    if((combos[0,1] - combos[0,0])%3 == (combos[1,1] - combos[1,0])%3):
        return False
    else:
        return True

def Orientate(mesh, DGnormal = None):
    ''' Orientates the given mesh with the given Normal, 
    '''
    mesh.init()
    #not parallel
    if DGnormal == None:
        if df.MPI.size(df.MPI.comm_world) > 1:
            print("automatric orientation is not possible in parallel. Provide a global normal vector!")
            exit()
        NumCells = mesh.num_cells()
        mesh.init()
        tdim = mesh.topology().dim()
        if tdim != 2 and df.MPI.rank(df.MPI.comm_world) == 0:
            print("Error: Expected topologically 2D Mesh for orientation")
            exit()
        orientation = df.MeshFunction('int', mesh, tdim)
        orientation.set_all(0)
        cell_neighbors = {cell.index(): sum((list(filter(lambda ci: ci != cell.index(),
                                                    facet.entities(tdim)))
                                            for facet in df.facets(cell)), [])
                        for cell in df.cells(mesh)}
        oriented = np.zeros(mesh.num_cells(),dtype = 'int64') # 1, if cell is oriented; 0, else
        oriented[0] = 1
        old = 0
        defects = np.zeros(mesh.num_cells(),dtype = 'int64')
        defect = False
        while np.amin(oriented)==0:
            if old == int(np.linalg.norm(oriented, 1)):
                defect = True
                print('    surface not connected')
                for i in range(np.size(oriented)):
                    if oriented[i]==0:
                        oriented[i] = 1
                        defects[i] = 1
                        break
            old = int(np.linalg.norm(oriented, 1))
            for i in range(np.size(oriented)):
                if oriented[i]==0: # trying to orient cell i now 
                    for j in range(len(cell_neighbors[i])):
                        if oriented[cell_neighbors[i][j]]==1:
                            c1 = df.Cell(mesh,i)
                            c2 = df.Cell(mesh,cell_neighbors[i][j])
                            same = IsSame(c1, c2)
                            if same:
                                orientation.array()[i] = orientation.array()[c2.index()]
                            else:
                                orientation.array()[i] = -orientation.array()[c2.index()]+1
                            oriented[i] = 1
                            if defect:
                                defects[i] = 1
                            break
        
        DGnormal = DGNormal(mesh, orientation)
        #return orientation
    #FIX FOR FENICS PRIVATE ORIENTATION CLASS:
    #Store the orientation in the fenics mesh class
    n = df.CellNormal(mesh)
    mu = df.FacetNormal(mesh)

    dofmap0 = DGnormal.function_space().sub(0).dofmap()
    dofmap1 = DGnormal.function_space().sub(1).dofmap()
    dofmap2 = DGnormal.function_space().sub(2).dofmap()


    CheckOrientation = df.assemble( (df.inner((n)('+'),(n)('-'))   +  df.inner(mu('+'), mu('-')))**2*df.dS(domain=mesh))
    #print("orientation correct? (0 means yes)", CheckOrientation)

    class MyGlobalNormal(df.UserExpression):
        def myinit(self, Factor):
            self.i = 0
            self.Factor = Factor
            self.normalvalues = DGnormal.vector().get_local()
        def eval(self, value, x):
            value[0] = self.Factor*self.normalvalues[dofmap0.cell_dofs(self.i)[0]]
            value[1] = self.Factor*self.normalvalues[dofmap1.cell_dofs(self.i)[0]]
            value[2] = self.Factor*self.normalvalues[dofmap2.cell_dofs(self.i)[0]]
            self.i+=1

        def value_shape(self):
            return (3,)

    myexp = MyGlobalNormal()
    myexp.myinit(1.0)
    mesh.init_cell_orientations(myexp)

    CheckOrientation = df.assemble( (df.inner((n)('+'),(n)('-'))   +  df.inner(mu('+'), mu('-')))**2*df.dS(domain=mesh))
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print("orientating worked? (0 means yes)", CheckOrientation)
    
    #test if normal is outward or inward:
    x = df.SpatialCoordinate(mesh)
    n = df.CellNormal(mesh)
    vol = df.assemble(df.Constant(1.0/3.0)*df.inner(x,n)*df.dx(domain=mesh))
    if vol < 0.0 :
        myexp = MyGlobalNormal()
        myexp.myinit(-1.0)
        mesh.init_cell_orientations(myexp)
    #double check
    vol = df.assemble(df.Constant(1.0/3.0)*df.inner(x,n)*df.dx(domain=mesh))
    if df.MPI.rank(df.MPI.comm_world) == 0:
        if vol < 0.0:
            print("Error during orientation")
            exit()
        else:
            print("Volume is positive")


def DGNormal(mesh, orientation = None):
    #turns the cell normal into a DG0 Function (On Boundary Mesh only)
    if mesh.topology().dim() == 3:
        print("Warning: BoundaryMesh is not oriented")
        gamma = df.BoundaryMesh(mesh, "exterior")
    else:
        gamma = mesh
        if orientation != None:
            if np.size(orientation.array()) != gamma.num_cells():
                print("Orientation array too short?!")
        else:
            if np.size(gamma.cell_orientations()) != gamma.num_cells():
                print("DGNormal: Mesh is not oriented and orientation is not provided")
                exit()
    MaxDim = gamma.geometry().dim()
    SpaceV = df.VectorFunctionSpace(gamma, "DG", 0, MaxDim)
    IntrinsicNormal = df.Function(SpaceV)
    len = IntrinsicNormal.vector().local_size()
    values = np.zeros(len)
    SpaceS = df.FunctionSpace(gamma,"DG", 0)
    dofs = SpaceS.dofmap()
    d2vector = (SpaceV.sub(0).dofmap().dofs(), SpaceV.sub(1).dofmap().dofs(), SpaceV.sub(2).dofmap().dofs())
    for c in df.cells(gamma):
        k = c.index()
        if orientation != None:
            Factor = -1.0
            Flip = orientation.array()[k]
            if Flip  > 0.5:
                Factor = 1.0
        else:
            if gamma.cell_orientations()[k] < 0.5:
                Factor = 1.0
            else:
                Factor = -1.0
        DofID = dofs.cell_dofs(k)[0]
        for i in range(MaxDim):
            values[d2vector[i][DofID]] = Factor*c.cell_normal()[i]
    IntrinsicNormal.vector().set_local(values)
    IntrinsicNormal.vector().apply("")
    IntrinsicNormal.rename("normal", "")
    return IntrinsicNormal
    
def ReadMeshH5(filename, closeFile=True):
    if os.path.splitext(filename)[1] == ".h5":
        s=filename
    else:
        s = filename + '.h5'
    hdf = df.HDF5File(df.MPI.comm_world, s,'r')
    mesh = df.Mesh()
    hdf.read(mesh, 'mesh', False)
    mesh.init()
    if hdf.has_dataset('normal'):
        vectorDG = df.VectorFunctionSpace(mesh, "DG", 0)
        DGnormal = df.Function(vectorDG)
        hdf.read(DGnormal,'normal')
        Orientate(mesh, DGnormal)
    else:
        print("mesh could not be orientated, no dataset \"normal\" included in H5")

    if closeFile:
        hdf.close()
        return mesh
    else:
        return mesh, hdf


def SaveMeshH5(mesh, filename , closeFile=True):
    orientation = np.asarray(mesh.cell_orientations(), dtype = 'float')
    if os.path.splitext(filename)[1] == ".h5":
        s=filename
    else:
        s = filename + '.h5'
    hdf = df.HDF5File(df.MPI.comm_world, s,'w')
    hdf.write(mesh,'mesh')
    vectorDG = df.VectorFunctionSpace(mesh, "DG", 0)
    DGnormal = df.project(df.CellNormal(mesh), vectorDG)
    hdf.write(DGnormal,'normal')
    if closeFile:
        hdf.close()
        return 
    else:
        return hdf

def ReadObj(ObjFile, PictureFile=None, IntOrder=0, Color = False):
    """ Reads 3D-scanner-data from a OBJ-File and (optional) corresponding texture stored in an Image

    Parameters
    -----------
    ObjFile: string
        the name of the OBJ-File
    PictureFile : string
        the name of the Image
    IntOrder : Int
        the order of the resulting DG function 
    Color : boolean, optional
        determines whether the texture image contains gray-scale or multi-channel pixel data, by default :code:`False`
    """
    
    #parallel
    #bjFile = './Input/ObjectFiles/' + ObjFile
    #if PictureFile != None:
        #PictureFile = './Input/ObjectFiles/' + PictureFile
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print("Reading %s"%ObjFile)
    with open(ObjFile, "r") as InFile:
        # read mesh vertices, textures, facets from the .obj file
        if df.MPI.rank(df.MPI.comm_world) == 0:
            print("    Reading Vertices, Textures and Facets...")
        NumVertex = 0
        NumTextures = 0
        NumFacets = 0
        for line in InFile:
            Ident = line[0:2]
            if Ident == "v ":
                NumVertex += 1
            if Ident == "vt":
                NumTextures += 1  
            if Ident == "f ":
                NumFacets += 1    
        InFile.seek(0)
        if df.MPI.rank(df.MPI.comm_world) == 0:
            print("    NumVertices %d"%NumVertex)
            print('    NumTextures %d'%NumTextures)
            print("    NumFacets %d"%NumFacets)
        MyVertices = np.zeros([NumVertex, 3])
        MyTextures = np.zeros([NumTextures, 2])
        MyFacets = np.zeros([NumFacets, 6], dtype="uintp")
        
        i = 0
        j = 0
        k = 0
        
        for line in InFile:
            Ident = line[0:2]
            if Ident == "v ":
                MySplit = line.split(" ")
                v1 = float(MySplit[1])
                v2 = float(MySplit[2])
                v3 = float(MySplit[3])
                MyVertices[i,0] = v1
                MyVertices[i,1] = v2
                MyVertices[i,2] = v3
                i += 1
                
            if Ident == "vt":
                MySplit = line.split(" ")
                t1 = float(MySplit[1])
                t2 = float(MySplit[2])
                MyTextures[j,0] = t1
                MyTextures[j,1] = t2
                j += 1
            
            if Ident == "f ":
                if PictureFile != None:
                    MySplit = line.split(" ")
                    e1 = int(MySplit[1].split("/")[0])
                    e2 = int(MySplit[2].split("/")[0])
                    e3 = int(MySplit[3].split("/")[0])
                    t1 = int(MySplit[1].split("/")[1])
                    t2 = int(MySplit[2].split("/")[1])
                    t3 = int(MySplit[3].split("/")[1])
                    MyFacets[k, 0] = e1-1
                    MyFacets[k, 1] = e2-1
                    MyFacets[k, 2] = e3-1
                    MyFacets[k, 3] = t1-1
                    MyFacets[k, 4] = t2-1
                    MyFacets[k, 5] = t3-1
                else:
                    MySplit = line.split(" ")
                    try:
                        e1 = int(MySplit[1])
                        e2 = int(MySplit[2])
                        e3 = int(MySplit[3])
                    except:
                        e1 = int(MySplit[1].split("//")[0])
                        e2 = int(MySplit[2].split("//")[0])
                        e3 = int(MySplit[3].split("//")[0])
                    MyFacets[k, 0] = e1-1
                    MyFacets[k, 1] = e2-1
                    MyFacets[k, 2] = e3-1
                k += 1
                
    #mesh
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print("    Generating mesh...")
    MyMesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(MyMesh, "triangle", 2, 3)
    editor.init_vertices_global(NumVertex, NumVertex)
    editor.init_cells_global(NumFacets, NumFacets)
    
    #build fenics mesh serial:
    if df.MPI.rank(df.MPI.comm_world) == 0:
        for i in range(NumVertex):
            editor.add_vertex_global(i,i, MyVertices[i, :])
        for k in range(NumFacets):
            editor.add_cell(k, MyFacets[k, 0:3])

    editor.close()

    #File("debug.pvd") << MyMesh
       
    df.MeshPartitioning.build_distributed_mesh(MyMesh)

    #File("xdebug.pvd") << MyMesh
        
    MyMesh.init()
    MyMesh.order()
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print("    Generating CoordinateFunction...")
    ##########################################################
    if PictureFile != None:

        CoordinateSpace = df.VectorFunctionSpace(MyMesh,"DG",1,2)
        CoordinateFunction = df.Function(CoordinateSpace)
        
        l_size = CoordinateFunction.vector().local_size()
        Fvalues = np.zeros(l_size)
        CDofs = df.FunctionSpace(MyMesh,"DG",1).dofmap()
        
        if Color:
            im = Image.open(PictureFile)
            ImArray = np.asarray(im, dtype='int64')
        else:
            im = Image.open(PictureFile).convert("L")
            ImArray = np.asarray(im, dtype='int64')
        for c in df.cells(MyMesh):
            
            k = c.index()

            DID1 = CDofs.cell_dofs(k)[0]
            DID2 = CDofs.cell_dofs(k)[1]
            DID3 = CDofs.cell_dofs(k)[2]

            k = c.global_index()

            c1 = int(round(MyTextures[MyFacets[k, 3],0]*np.size(ImArray,1), 0))
            c2 = int(round(MyTextures[MyFacets[k, 3],1]*np.size(ImArray,0), 0))
            c3 = int(round(MyTextures[MyFacets[k, 4],0]*np.size(ImArray,1), 0))
            c4 = int(round(MyTextures[MyFacets[k, 4],1]*np.size(ImArray,0), 0))
            c5 = int(round(MyTextures[MyFacets[k, 5],0]*np.size(ImArray,1), 0))
            c6 = int(round(MyTextures[MyFacets[k, 5],1]*np.size(ImArray,0), 0))
            
            if np.linalg.norm(c.get_vertex_coordinates()[0:3]-MyVertices[MyFacets[k, 0],0:3])<0.000001:
                Fvalues[2*DID1+0]=np.size(ImArray,0)-c2
                Fvalues[2*DID1+1]=c1
                if np.linalg.norm(c.get_vertex_coordinates()[3:6]-MyVertices[MyFacets[k, 1],0:3])<0.000001:
                    Fvalues[2*DID2+0]=np.size(ImArray,0)-c4
                    Fvalues[2*DID2+1]=c3
                    Fvalues[2*DID3+0]=np.size(ImArray,0)-c6
                    Fvalues[2*DID3+1]=c5
                else:
                    Fvalues[2*DID3+0]=np.size(ImArray,0)-c4
                    Fvalues[2*DID3+1]=c3
                    Fvalues[2*DID2+0]=np.size(ImArray,0)-c6
                    Fvalues[2*DID2+1]=c5
            elif np.linalg.norm(c.get_vertex_coordinates()[0:3]-MyVertices[MyFacets[k, 1],0:3])<0.000001:
                Fvalues[2*DID1+0]=np.size(ImArray,0)-c4
                Fvalues[2*DID1+1]=c3
                if np.linalg.norm(c.get_vertex_coordinates()[3:6]-MyVertices[MyFacets[k, 0],0:3])<0.000001:
                    Fvalues[2*DID2+0]=np.size(ImArray,0)-c2
                    Fvalues[2*DID2+1]=c1
                    Fvalues[2*DID3+0]=np.size(ImArray,0)-c6
                    Fvalues[2*DID3+1]=c5
                else:
                    Fvalues[2*DID2+0]=np.size(ImArray,0)-c6
                    Fvalues[2*DID2+1]=c5
                    Fvalues[2*DID3+0]=np.size(ImArray,0)-c2
                    Fvalues[2*DID3+1]=c1
            else:
                Fvalues[2*DID1+0]=np.size(ImArray,0)-c6
                Fvalues[2*DID1+1]=c5
                if np.linalg.norm(c.get_vertex_coordinates()[3:6]-MyVertices[MyFacets[k, 0],0:3])<0.000001:
                    Fvalues[2*DID2+0]=np.size(ImArray,0)-c2
                    Fvalues[2*DID2+1]=c1
                    Fvalues[2*DID3+0]=np.size(ImArray,0)-c4
                    Fvalues[2*DID3+1]=c3
                else:
                    Fvalues[2*DID2+0]=np.size(ImArray,0)-c4
                    Fvalues[2*DID2+1]=c3
                    Fvalues[2*DID3+0]=np.size(ImArray,0)-c2
                    Fvalues[2*DID3+1]=c1     
        CoordinateFunction.vector().set_local(Fvalues)
        CoordinateFunction.vector().apply("")

        TextureOrder = IntOrder
        
        if TextureOrder != 1:
            if df.MPI.rank(df.MPI.comm_world) == 0:
                print("    Interpolating CoordinateFunction...")
            CoordinateFunction = df.interpolate(CoordinateFunction, df.VectorFunctionSpace(MyMesh, "DG", TextureOrder, 2))
        if df.MPI.rank(df.MPI.comm_world) == 0:
            print("    Generating TextureFunction...")
        
        if Color:
            TextureFunction = df.Function(df.VectorFunctionSpace(MyMesh, "DG", TextureOrder,3))
            loc_size = int(TextureFunction.vector().local_size()/3.0)
        else:
            TextureFunction = df.Function(df.FunctionSpace(MyMesh, "DG", TextureOrder))
            loc_size = TextureFunction.vector().local_size()
        TextureFunction.rename("Texture", "")
        FValues = np.zeros(TextureFunction.vector().local_size())

        if Color:
            for i in range(loc_size):
                TX = int(round(np.abs(CoordinateFunction.vector()[2*i+0]-0.5),1))
                TY = int(round(np.abs(CoordinateFunction.vector()[2*i+1]-0.5),1))
                FValues[3*i+0] = ImArray[TX, TY, 0]/255.0
                FValues[3*i+1] = ImArray[TX, TY, 1]/255.0
                FValues[3*i+2] = ImArray[TX, TY, 2]/255.0
        else:
            for i in range(loc_size):
                TX = int(round(np.abs(CoordinateFunction.vector()[2*i+0]-0.5),1))
                TY = int(round(np.abs(CoordinateFunction.vector()[2*i+1]-0.5),1))
                FValues[i] = ImArray[TX, TY]/255.0
            
        TextureFunction.vector().set_local(FValues)
        TextureFunction.vector().apply("")
        if df.MPI.rank(df.MPI.comm_world) == 0:
            print('Reading done.')
    else: 
        TextureFunction = df.Function(df.FunctionSpace(MyMesh, "DG", IntOrder))
        
    
    return TextureFunction


def CGdofstoEdgemarker(mesh, CGfun, dofmap):
    boundary_marker = df.MeshFunction('size_t', mesh, 1)
    boundary_marker.set_all(0)
    vec = CGfun.vector().get_local()
    for e in df.edges(mesh):
        dofs = dofmap.entity_dofs(mesh, 0, e.entities(0))
        if vec[dofs[0]] < 0.5 and vec[dofs[1]] < 0.5:
            boundary_marker[e.index()] = 1
    return boundary_marker



def repairEdges(mesh,n):
    spaceE = df.FunctionSpace(mesh, "HDiv Trace", 0)    
    CG = df.VectorFunctionSpace(mesh, "CG", 1)
    
    
    d = df.Function(CG)
    dvec = d.vector().get_local()
    tmp = df.assemble( df.inner(n('+'), n('-'))*df.TestFunction(spaceE)('+')/df.FacetArea(mesh)('+')*df.dS).get_local()
    
    for e in df.edges(mesh):
        if tmp[spaceE.dofmap().entity_dofs(mesh,1,[e.index()])[0]] < -np.sqrt(0.5):
            #fix edge
            c0 = e.entities(2)[0]
            c1 = e.entities(2)[1]
            v0 = sum(df.Cell(mesh, c0).entities(0))-sum(e.entities(0))
            v1 = sum(df.Cell(mesh, c1).entities(0))-sum(e.entities(0))
            p0 =  np.array([0.,0.,0.])
            n0 = 0
            for c in df.Vertex(mesh,v0).entities(2):
                for v in df.Cell(mesh, c).entities(0):
                    if v != v0:
                        vert = df.Vertex(mesh,v)
                        p0 += vert.midpoint().array()
                        n0 += 1
            p1 = np.array([0.,0.,0.])
            n1 = 0
            for c in df.Vertex(mesh,v1).entities(2):
                for v in df.Cell(mesh, c).entities(0):
                    if v != v1:
                        vert = df.Vertex(mesh,v)
                        p1 += vert.midpoint().array()
                        n1 += 1
            
            #from IPython import embed; embed()
            dvec[CG.sub(0).dofmap().entity_dofs(mesh,0,[v0])[0]] = p0[0]/n0-df.Vertex(mesh,v0).midpoint().array()[0]
            dvec[CG.sub(1).dofmap().entity_dofs(mesh,0,[v0])[0]] = p0[1]/n0-df.Vertex(mesh,v0).midpoint().array()[1]
            dvec[CG.sub(2).dofmap().entity_dofs(mesh,0,[v0])[0]] = p0[2]/n0-df.Vertex(mesh,v0).midpoint().array()[2]
            dvec[CG.sub(0).dofmap().entity_dofs(mesh,0,[v1])[0]] = p1[0]/n1-df.Vertex(mesh,v1).midpoint().array()[0]
            dvec[CG.sub(1).dofmap().entity_dofs(mesh,0,[v1])[0]] = p1[1]/n1-df.Vertex(mesh,v1).midpoint().array()[1]
            dvec[CG.sub(2).dofmap().entity_dofs(mesh,0,[v1])[0]] = p1[2]/n1-df.Vertex(mesh,v1).midpoint().array()[2]
    d.vector().set_local(dvec)
    df.ALE.move(mesh, d)
            #d.vector().set_local(d.vector().get_local()*0) 
            #tmp = df.assemble( df.inner(n('+'), n('-'))*df.TestFunction(spaceE)('+')/df.FacetArea(mesh)('+')*df.dS).get_local()
    pass
    
    