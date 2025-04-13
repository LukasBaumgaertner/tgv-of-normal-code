import dolfin as df
df.set_log_level(30)
from utils.shape_identities import log, dlog, ddlog, divE, Hedge, Hsurf, tan_jump, dmu, ddmu, dtE, ddtE, dn, ddn, parallelTransportTensor, parallelTransport, dparallelTransport, ddparallelTransport, logmu, dlogmu, ddlogmu
import utils.utils
import numpy as np
import ufl
import time
import truncatedCG
import sys
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

import petsc4py.PETSc as PETSc
import time


#df.parameters["reorder_dofs_serial"] = True

class AugmentedLagrangian():
    def __init__(self, mesh, rhos=[1,1,1,1], alphas=[1,1], tau=1e-7, tracked_vertices = None, fixed_vertices = None, adapt_penalty = True):
        

        
        # mesh stuff
        self.mesh = mesh
        self.n = df.CellNormal(mesh)
        self.mu = df.FacetNormal(mesh)
        self.tE = df.cross(self.n,self.mu)
        
        # saving parameters, rho potentially changes but we dont want to redefine the forms and make it a function
        R = df.FunctionSpace(mesh, "R", 0)
        self.rho = [df.project(df.Constant(rho), R) for rho in rhos]
        self.tau = df.Constant(tau)
        self.alpha = [df.Constant(alpha) for alpha in alphas]
        
        #deformation space
        self.CG = df.VectorFunctionSpace(mesh, "CG", 1)
        self.DGvec = df.VectorFunctionSpace(mesh, "DG", 0)
        self.DGscalar = df.FunctionSpace(mesh, "DG", 0)

        #test and trial functinos for shape derivatives
        self.V = df.TestFunction(self.CG)
        self.W = df.TrialFunction(self.CG)
        
        
        # elements of the tangential RT are resresented in a DRT space, the two dofs are saved in a seperate function
        # changing the mesh requires recomputation from the dofs 
        self.RT = df.VectorFunctionSpace(mesh, "DRT", 1)
        self.RTdof = df.VectorFunctionSpace(mesh, "HDiv Trace", 0, dim=2)
        
        # auxiliary variable is optimized
        self.with_aux = True
        self.adapt_penalty = True
        
  
        self.u = df.Function(self.RT)
        self.du = df.Function(self.RT)
        self.utmp = df.Function(self.RT)
        self.utmp1 = df.Function(self.RT)
        self.utmp2 = df.Function(self.RT)
        self.udof = df.Function(self.RTdof)
        self.dudof = df.Function(self.RTdof)
        self.d_AL_aux_assembled = df.Function(self.RT) 
        
        self.n_0 = df.project(self.n, self.DGvec)
        
        
        
        self.v = df.TestFunction(self.RT)
        self.vdof = df.TestFunction(self.RTdof)
        
        self.w = df.TrialFunction(self.RT)
        self.wdof = df.TrialFunction(self.RTdof)


        # movefun to move the mesh later
        self.movefun = df.Function(self.CG)
        # help function for shape gradient
        self.shapegrad = df.Function(self.CG)
        # scalar function CG1
        self.scalarCG = df.FunctionSpace(self.mesh, "CG", 1)
        self.normu = df.Function(df.FunctionSpace(self.mesh, "DG",0))
        
        #save tracking
        self.x = df.SpatialCoordinate(self.mesh)
        self.xdir = df.Function(self.CG)
        self.tracking_data = df.project(self.x, self.CG)
        
        # spaces auxiliary variables
        self.skeletonDG = df.FunctionSpace(mesh, "HDiv Trace", 0)
        self.skeletonDGvec = df.VectorFunctionSpace(mesh, "HDiv Trace", 0, dim = 3)
        self.skeletonDGveclin = df.VectorFunctionSpace(mesh, "HDiv Trace", 1, dim = 3)
        self.skeletonDGquad = df.FunctionSpace(mesh, "HDiv Trace",2)
        self.DGten = df.TensorFunctionSpace(mesh, "DG", 0, shape = (3,3,3))
        # auxiliary variable and multiplier
        self.d = [df.Function(self.skeletonDG),df.Function(self.DGten), df.Function(self.skeletonDGveclin) ]
        self.b = [df.Function(self.skeletonDG),df.Function(self.DGten), df.Function(self.skeletonDGveclin) ]
        
        
        # define two polynomials which define the endpointwise inner produdct on linear functions on edges 
        self.orth1 = df.Function(self.skeletonDGquad)
        self.orth2 = df.Function(self.skeletonDGquad)
        dofmap = self.skeletonDGquad.dofmap()
        dofmaplin = self.skeletonDGveclin.dofmap()
        vecorth1 = self.orth1.vector().get_local()
        vecorth2 = self.orth2.vector().get_local()
        for e in df.edges(mesh):
            dofs= dofmap.entity_dofs(mesh, 1, [e.index()])
            
            if len(dofs) != 3:
                print("error")
            vecorth1[dofs[0]] = 9
            vecorth1[dofs[1]] = 3
            vecorth1[dofs[2]] = -1.5
            
            
            vecorth2[dofs[0]] = 3
            vecorth2[dofs[1]] = 9
            vecorth2[dofs[2]] = -1.5
        self.orth1.vector().set_local(vecorth1)
        self.orth2.vector().set_local(vecorth2)
        
        
        #assembling returns the dof vector for the discontinious RT from the current self.udof function 
        self.dofform = self.udof[0]('+')*df.inner(self.mu, df.dot(self.v, self.mu))('+')*df.dS 
        self.dofform += self.udof[1]('+')*df.inner(self.tE('+'), df.dot(self.v, self.mu)('+'))*df.dS
        self.dofform += self.udof[0]('+')*df.inner(self.mu, df.dot(self.v, self.mu))('-')*df.dS 
        self.dofform += -self.udof[1]('+')*df.inner(self.tE('+'), df.dot(self.v, self.mu)('-'))*df.dS 
        self.u.vector().set_local( df.assemble(self.dofform))

        # assembleling returns a matrix which computs the dof vector for the discontinious RT from a dof vectors of the tangent RT
        self.dofmapform =  self.vdof[0]('+')*df.inner(self.mu, df.dot(self.w, self.mu))('+')*df.dS 
        self.dofmapform += self.vdof[0]('+')*df.inner(self.mu, df.dot(self.w, self.mu))('-')*df.dS 
        self.dofmapform += self.vdof[1]('+')*df.inner(self.tE('+'), df.dot(self.w, self.mu)('+'))*df.dS 
        self.dofmapform += -self.vdof[1]('+')*df.inner(self.tE('+'), df.dot(self.w, self.mu)('-'))*df.dS 
         
        # same as above but transposed
        self.dofmapformT =  self.wdof[0]('+')*df.inner(self.mu, df.dot(self.v, self.mu))('+')*df.dS 
        self.dofmapformT += self.wdof[0]('+')*df.inner(self.mu, df.dot(self.v, self.mu))('-')*df.dS 
        self.dofmapformT += self.wdof[1]('+')*df.inner(self.tE('+'), df.dot(self.v, self.mu)('+'))*df.dS 
        self.dofmapformT += -self.wdof[1]('+')*df.inner(self.tE('+'), df.dot(self.v, self.mu)('-'))*df.dS 
        
        self.dofmapmat = df.assemble(self.dofmapform)
        


        #compute the shape derivative with respect to the (DRT function) u when the same dofs for the TangentRT are used on the perturbed mesh 
        #dof1 = |E|<mu, u@mu> = <mu, dofd>  -> 0 = <dmu[V],dofd> + <mu, d(dofd)[V]> <->  <mu, d(dofd)[V]> = -<dmu[V],dofd>
        #dof2 = |E|<tE, u@mu> = <tE, dofd>  -> 0 = <dtE[V],dofd> + <tE, d(dofd)[V]> <->  <tE, d(dofd)[V]> = -<dtE[V],dofd>
        #dof2 = -|E|<tE, u@mu> = -<tE, dofd>  -> 0 = <dtE[V],dofd> + <tE, d(dofd)[V]> <->  <tE, d(dofd)[V]> = -<dtE[V],dofd>
        # dofd = mu*dof1 + tE*dof2
        
        dofd_plus = self.mu('+')*self.udof[0]('+') + self.tE('+')*self.udof[1]('+') 
        dofd_minus = self.mu('-')*self.udof[0]('+') - self.tE('+')*self.udof[1]('+') 
        
        #assembling returns a matrix that converts a shape pertubation into a perturbation on u in DRT (when the same dofs are used for tangentRT)
        self.d_dofform = -df.inner( dmu(self.mu('+'), self.tE('+'), self.V('+') ), dofd_plus)*df.inner(self.mu('+'), df.dot(self.w, self.mu)('+'))*df.dS 
        self.d_dofform += -df.inner( dtE(self.tE('+'), self.V('+') ), dofd_plus )*df.inner(self.tE('+'), df.dot(self.w, self.mu)('+'))*df.dS       
        self.d_dofform += -df.inner( dn(self.n('+'), self.V('+') ), dofd_plus )*df.inner(self.n('+'), df.dot(self.w, self.mu)('+'))*df.dS       
        self.d_dofform += -df.inner( dmu(self.mu('-'), self.tE('+'), self.V('-') ),  dofd_minus )*df.inner(self.mu('-'), df.dot(self.w, self.mu)('-'))*df.dS 
        self.d_dofform += -df.inner( dtE(self.tE('+'), self.V('-') ), dofd_minus )*df.inner(self.tE('+'), df.dot(self.w, self.mu)('-'))*df.dS 
        self.d_dofform += -df.inner( dn(self.n('-'), self.V('-') ), dofd_minus )*df.inner(self.n('-'), df.dot(self.w, self.mu)('-'))*df.dS 
 
        # transposed matrix of the above
        self.d_dofformT = -df.inner( dmu(self.mu('+'), self.tE('+'), self.W('+') ), dofd_plus)*df.inner(self.mu('+'), df.dot(self.v, self.mu)('+'))*df.dS 
        self.d_dofformT += -df.inner( dtE(self.tE('+'), self.W('+') ), dofd_plus )*df.inner(self.tE('+'), df.dot(self.v, self.mu)('+'))*df.dS       
        self.d_dofformT += -df.inner( dn(self.n('+'), self.W('+') ), dofd_plus )*df.inner(self.n('+'), df.dot(self.v, self.mu)('+'))*df.dS       
        self.d_dofformT += -df.inner( dmu(self.mu('-'), self.tE('+'), self.W('-') ),  dofd_minus )*df.inner(self.mu('-'), df.dot(self.v, self.mu)('-'))*df.dS 
        self.d_dofformT += -df.inner( dtE(self.tE('+'), self.W('-') ), dofd_minus )*df.inner(self.tE('+'), df.dot(self.v, self.mu)('-'))*df.dS 
        self.d_dofformT += -df.inner( dn(self.n('-'), self.W('-') ), dofd_minus )*df.inner(self.n('-'), df.dot(self.v, self.mu)('-'))*df.dS 

 
        self.d_dofform_aux_assembled = self.d_dofform*self.d_AL_aux_assembled
        self.d_dofform_utmp = self.d_dofform*self.utmp
        
        
        
        
        # Computing the second shape derivative of the funtion u in DRT (when the same dofs are used for the tangentRT function)
        # dof1 = |E|<mu, u@mu> , dof2 = |E|<tE, u@mu>
        # 0 = <dmu[V], dofd> + <mu, ddofd[V]> , 0 = <dtE[V], dofd> + <tE, ddofd[V]> 
        # ddof[V] = -mu*<dmu[V], dofd> - tE*<dtE[V], dofd>
        
        
        # 0 = <ddmu[V,W], dofd> + <dmu[V], ddofd[W]> + <dmu[W], ddofd[V]> + <mu, dddofd[V,W]>
        # <mu, dddofd[V,W]> = - <ddmu[V,W], dofd> + <dmu[V],  mu*<dmu[W], dofd> + tE*<dtE[W], dofd> > + <dmu[W], mu*<dmu[V], dofd> + tE*<dtE[V], dofd> > 
        # <dmu[V], mu> = 0 yields
        # <mu, dddofd[V,W]> = - <ddmu[V,W], dofd> + <dmu[V], tE*<dtE[W], dofd> > + <dmu[W], tE*<dtE[V], dofd> > 
        # <tE, dddofd[V,W]> = - <ddtE[V,W], dofd> + <dtE[V],  mu*<dmu[W], dofd> + tE*<dtE[W], dofd> > + <dtE[W], mu*<dmu[V], dofd> + tE*<dtE[V], dofd> > 
        # <dtE[V], tE> = 0 yields
        # <tE, dddofd[V,W]> = - <ddtE[V,W], dofd> + <dtE[V],  mu*<dmu[W], dofd> > + <dtE[W], mu*<dmu[V], dofd>  > 
        
        # no normal component:
        # 0 = |E|<n, u@mu> -> 0 = <dn[V], dofd> + <n, ddofd[V]> or <n, ddofd[V]> = - <dn[V], dofd>
       
        # <mu, dddofd[V,W]> = - <ddmu[V,W], dofd> + <dmu[V], tE*<dtE[W], dofd> + n*<dn[W], dofd> > + <dmu[W], tE*<dtE[V], dofd> + n*<dn[W], dofd> > 
        # <n, dddofd[V,W]> = - <ddn[V,W], dofd> + <dn[V], mu*<dmu[W], dofd> + tE*<dtE[W], dofd> > + <dn[W], mu*<dmu[V], dofd> + tE*<dtE[W], dofd> > 
      
        
        self.H_dofform = -df.inner( ddmu(self.mu('+'), self.tE('+'), self.V('+'), self.W('+') ), dofd_plus )*df.inner(self.mu('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dmu(self.mu('+'), self.tE('+'), self.V('+')), self.tE('+')*df.inner(dtE(self.tE('+'), self.W('+')),dofd_plus) )  *df.inner(self.mu('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dmu(self.mu('+'), self.tE('+'), self.V('+')), self.n('+')*df.inner(dn(self.n('+'), self.W('+')),dofd_plus) )  *df.inner(self.mu('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dmu(self.mu('+'), self.tE('+'), self.W('+')), self.tE('+')*df.inner(dtE(self.tE('+'), self.V('+')),dofd_plus) )  *df.inner(self.mu('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dmu(self.mu('+'), self.tE('+'), self.W('+')), self.n('+')*df.inner(dn(self.n('+'), self.V('+')),dofd_plus) )  *df.inner(self.mu('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        
        self.H_dofform += -df.inner( ddtE(self.tE('+'), self.V('+'), self.W('+') ), dofd_plus )*df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dtE(self.tE('+'), self.V('+')), self.mu('+')*df.inner(dmu(self.mu('+'), self.tE('+'), self.W('+')),dofd_plus) )  *df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dtE(self.tE('+'), self.V('+')), self.n('+')*df.inner(dn(self.n('+'),self.W('+')),dofd_plus) )  *df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dtE(self.tE('+'), self.W('+')), self.mu('+')*df.inner(dmu(self.mu('+'), self.tE('+'), self.V('+')),dofd_plus) )  *df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dtE(self.tE('+'), self.W('+')), self.n('+')*df.inner(dn(self.n('+'), self.V('+')),dofd_plus) )  *df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
     
        self.H_dofform += -df.inner( ddn(self.n('+'), self.V('+'), self.W('+') ), dofd_plus )*df.inner(self.n('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dn(self.n('+'), self.V('+')), self.mu('+')*df.inner(dmu(self.mu('+'), self.tE('+'), self.W('+')),dofd_plus) )  *df.inner(self.n('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dn(self.n('+'), self.V('+')), self.tE('+')*df.inner(dtE(self.tE('+'),self.W('+')),dofd_plus) )  *df.inner(self.n('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dn(self.n('+'), self.W('+')), self.mu('+')*df.inner(dmu(self.mu('+'), self.tE('+'), self.V('+')),dofd_plus) )  *df.inner(self.n('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        self.H_dofform += df.inner( dn(self.n('+'), self.W('+')), self.tE('+')*df.inner(dtE(self.tE('+'), self.V('+')),dofd_plus) )  *df.inner(self.n('+'), df.dot(self.d_AL_aux_assembled, self.mu)('+'))*df.dS 
        
            
        self.H_dofform += -df.inner( ddmu(self.mu('-'), self.tE('+'), self.V('-'), self.W('-') ), dofd_minus )*df.inner(self.mu('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dmu(self.mu('-'), self.tE('+'), self.V('-')), self.tE('+')*df.inner(dtE(self.tE('+'), self.W('-')),dofd_minus) )  *df.inner(self.mu('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dmu(self.mu('-'), self.tE('+'), self.V('-')), self.n('-')*df.inner(dn(self.n('-'), self.W('-')),dofd_minus) )  *df.inner(self.mu('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dmu(self.mu('-'), self.tE('+'), self.W('-')), self.tE('+')*df.inner(dtE(self.tE('+'), self.V('-')),dofd_minus) )  *df.inner(self.mu('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dmu(self.mu('-'), self.tE('+'), self.W('-')), self.n('-')*df.inner(dn(self.n('-'), self.V('-')),dofd_minus) )  *df.inner(self.mu('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        
        self.H_dofform += -df.inner( ddtE(self.tE('+'), self.V('-'), self.W('-') ), dofd_minus )*df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dtE(self.tE('+'), self.V('-')), self.mu('-')*df.inner(dmu(self.mu('-'), self.tE('+'), self.W('-')),dofd_minus) )  *df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dtE(self.tE('+'), self.V('-')), self.n('-')*df.inner(dn(self.n('-'), self.W('-')),dofd_minus) )  *df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dtE(self.tE('+'), self.W('-')), self.mu('-')*df.inner(dmu(self.mu('-'), self.tE('+'), self.V('-')),dofd_minus) )  *df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dtE(self.tE('+'), self.W('-')), self.n('-')*df.inner(dn(self.n('-'), self.V('-')),dofd_minus) )  *df.inner(self.tE('+'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
    
        self.H_dofform += -df.inner( ddn(self.n('-'), self.V('-'), self.W('-') ), dofd_minus )*df.inner(self.n('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dn(self.n('-'), self.V('-')), self.mu('-')*df.inner(dmu(self.mu('-'), self.tE('+'), self.W('-')),dofd_minus) )  *df.inner(self.n('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dn(self.n('-'), self.V('-')), self.tE('+')*df.inner(dtE(self.tE('+'), self.W('-')),dofd_minus) )  *df.inner(self.n('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dn(self.n('-'), self.W('-')), self.mu('-')*df.inner(dmu(self.mu('-'), self.tE('+'), self.V('-')),dofd_minus) )  *df.inner(self.n('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS 
        self.H_dofform += df.inner( dn(self.n('-'), self.W('-')), self.tE('+')*df.inner(dtE(self.tE('+'), self.V('-')),dofd_minus) )  *df.inner(self.n('-'), df.dot(self.d_AL_aux_assembled, self.mu)('-'))*df.dS         
        


        # computes curcumcenters from vertex positions, in particular, computes a matrix for this
        DGel = df.VectorElement("DG", mesh.ufl_cell(), 0)
        HDivel = df.VectorElement("HDiv Trace", mesh.ufl_cell(), 0)
        self.CCSpace = df.FunctionSpace(mesh, df.MixedElement([DGel, DGel, DGel,HDivel]))
        
        self.verts=df.Function(self.CCSpace)
        self.dverts=df.Function(self.CCSpace)
        self.vertstmp=df.Function(self.CCSpace)
        # this matrix will map the vector CG1 vertex dofs to DG0 Vertex dofs, thereby the vertex positions can be used inside the triangle
        # this also maps the midpoint of the edge to an vector HDiv trace space
        CCmat = lil_matrix((self.verts.vector().get_local().size, 3*mesh.num_vertices() ))
        
        for c in df.cells(mesh):
            vertAdofs = self.CG.dofmap().entity_dofs(mesh, 0, [c.entities(0)[0]])
            vertBdofs = self.CG.dofmap().entity_dofs(mesh, 0, [c.entities(0)[1]])
            vertCdofs = self.CG.dofmap().entity_dofs(mesh, 0, [c.entities(0)[2]])
            
            cdofsA = self.CCSpace.sub(0).dofmap().entity_dofs(mesh,2,[c.index()])
            cdofsB = self.CCSpace.sub(1).dofmap().entity_dofs(mesh,2,[c.index()])
            cdofsC = self.CCSpace.sub(2).dofmap().entity_dofs(mesh,2,[c.index()])
            
            
            CCmat[cdofsA[0], vertAdofs[0]] = 1
            CCmat[cdofsA[1], vertAdofs[1]] = 1
            CCmat[cdofsA[2], vertAdofs[2]] = 1
            
            CCmat[cdofsB[0], vertBdofs[0]] = 1
            CCmat[cdofsB[1], vertBdofs[1]] = 1
            CCmat[cdofsB[2], vertBdofs[2]] = 1
            
            CCmat[cdofsC[0], vertCdofs[0]] = 1
            CCmat[cdofsC[1], vertCdofs[1]] = 1
            CCmat[cdofsC[2], vertCdofs[2]] = 1
            
        for e in df.edges(mesh):
            vert1dofs = self.CG.dofmap().entity_dofs(mesh, 0, [e.entities(0)[0]])
            vert2dofs = self.CG.dofmap().entity_dofs(mesh, 0, [e.entities(0)[1]])
            
            edofs = self.CCSpace.sub(3).dofmap().entity_dofs(mesh,1,[e.index()])
            
            CCmat[edofs[0],vert1dofs[0]] = 0.5
            CCmat[edofs[0],vert2dofs[0]] = 0.5
            
            CCmat[edofs[1],vert1dofs[1]] = 0.5
            CCmat[edofs[1],vert2dofs[1]] = 0.5
            
            CCmat[edofs[2],vert1dofs[2]] = 0.5
            CCmat[edofs[2],vert2dofs[2]] = 0.5
        
        self.CCmat = csr_matrix(CCmat)
        self.set_cc_vecs(self.get_mesh_coords())
        
        
        
        self.vertA,self.vertB,self.vertC,self.vertM = df.split(self.verts)

        self.verta = self.vertA - self.vertC
        self.vertb = self.vertB - self.vertC

        # formular for the circumcenters
        crossab = df.cross(self.verta, self.vertb)
        self.cc = self.vertC + df.cross(df.inner(self.verta,self.verta)*self.vertb-df.inner(self.vertb,self.vertb)*self.verta, crossab)/2/df.inner(crossab, crossab)

        # h and its derivative
        self.h = df.inner(self.vertM('+') - self.cc('+'), self.mu('+')) + df.inner(self.vertM('+') - self.cc('-'), self.mu('-'))
        self.dh = lambda V: df.inner(self.vertM('+') - self.cc('+'), dmu( self.mu('+'), self.tE('+'), V('+'))) + df.inner(self.vertM('+') - self.cc('-'), dmu( self.mu('-'), self.tE('+'), V('-')))
        self.ddh = df.inner(self.vertM('+') - self.cc('+'), ddmu(self.mu('+'), self.tE('+'), self.V('+'), self.W('+')) ) + df.inner(self.vertM('+') - self.cc('-'), ddmu(self.mu('-'), self.tE('+'), self.V('-'), self.W('-')))

        # manuel derivative for the alpha_1 term
        self.logterm =  logmu(self.n, self.mu)  - df.inner(self.mu('+'), self.h*df.dot(self.u, self.mu)('+')) 
        self.dlogterm = lambda V: dlogmu(self.n, self.mu, V)  - df.inner(dmu(self.mu, self.tE, V)('+'), self.h*df.dot(self.u, self.mu)('+')) - df.inner(self.mu('+'), self.dh(V)*df.dot(self.u, self.mu)('+')) - df.inner(self.mu('+'), self.h*(-divE(self.tE,V))*df.dot(self.u, self.mu)('+'))
        self.ddlogterm = ddlogmu(self.n, self.mu, self.tE, self.V, self.W) 
        self.ddlogterm += -df.inner(ddmu(self.mu, self.tE, self.V, self.W)('+'), self.h*df.dot(self.u, self.mu)('+'))
        self.ddlogterm += -df.inner(dmu(self.mu, self.tE, self.W)('+'), self.dh(self.V)*df.dot(self.u, self.mu)('+')) - df.inner(dmu(self.mu, self.tE, self.W)('+'), self.h*(-divE(self.tE,self.V))*df.dot(self.u, self.mu)('+')) 
        self.ddlogterm += -df.inner(dmu(self.mu, self.tE, self.V)('+'), self.dh(self.W)*df.dot(self.u, self.mu)('+')) - df.inner(dmu(self.mu, self.tE, self.V)('+'), self.h*(-divE(self.tE,self.W))*df.dot(self.u, self.mu)('+')) 
        self.ddlogterm += - df.inner(self.mu('+'), self.dh(self.V)*(-divE(self.tE,self.W))*df.dot(self.u, self.mu)('+'))  - df.inner(self.mu('+'), self.dh(self.W)*(-divE(self.tE,self.V))*df.dot(self.u, self.mu)('+')) 
        self.ddlogterm += - df.inner(self.mu('+'), self.ddh*df.dot(self.u, self.mu)('+'))
        self.ddlogterm += - df.inner(self.mu('+'), self.h*(-Hedge(self.tE,self.V,self.W))*df.dot(self.u, self.mu)('+'))
        self.ddlogterm += - df.inner(self.mu('+'), self.h*(2*divE(self.tE,self.W)*divE(self.tE,self.V))*df.dot(self.u, self.mu)('+'))
        

        # form for non-augmented Lagrangian, required for residuals
        # fidelity
        self.L_form = 0.5*(self.x - self.tracking_data)**2*df.dP
        # term to prevent mesh degeneration
        self.L_form += self.tau*1/df.CellVolume(mesh)**2*df.dx(domain=mesh)
        # contraint
        self.L_form += self.rho[0]*self.b[0]('+')*( self.logterm - self.d[0]('+'))*df.dS(domain=mesh) 
        
        self.L_form += self.rho[1]*df.inner( self.b[1], self.grad(self.u)-self.d[1] )*df.dx
        
        self.L_form += self.rho[2]*df.inner(self.b[2]('+'), tan_jump(self.u, self.tE, self.mu) - self.d[2]('+')  )*(self.orth1('+')+self.orth2('+'))/2*df.dS
        
        self.d_L_form_aux = df.derivative(self.L_form, self.u, df.TestFunction(self.RT))


        # form for augmented Lagrangian, is seperated in terms to automatically differentiate and manuelly differentiate
        # fidelity
        self.AL_form_man = 0.5*(self.x - self.tracking_data)**2*df.dP
        # term to prevent mesh degeneration
        self.AL_form_man += self.tau*1/df.CellVolume(mesh)**2*df.dx(domain=mesh)
        # contraint
        self.AL_form_man += 0.5*self.rho[0]*(self.d[0]('+') - self.logterm - self.b[0]('+'))**2*df.dS(domain=mesh) 
        self.AL_form_man += -0.5*self.rho[0]*self.b[0]('+')**2*df.dS(domain=mesh)
        
        self.AL_form_auto = 0.5*self.rho[1]*( parallelTransportTensor(self.d[1]-self.b[1],self.n_0, self.n) - self.grad(self.u) )**2*df.dx
        self.AL_form_auto += -0.5*self.rho[1]*( self.b[1] )**2*df.dx
        
        self.AL_form_auto += 0.5*self.rho[2]*( parallelTransport(self.d[2]-self.b[2], self.n_0, self.n)('+') - tan_jump(self.u, self.tE, self.mu))**2*(self.orth1('+')+self.orth2('+'))/2*df.dS
        self.AL_form_auto += -0.5*self.rho[2]*( self.b[2]('+') )**2*(self.orth1('+')+self.orth2('+'))/2*df.dS
        
        self.AL_form = self.AL_form_man + self.AL_form_auto

        # derivative of AL 
        # fidelity
        self.d_AL_form_man = df.inner(self.x-self.tracking_data, self.V)*df.dP
        
        # term to prevent mesh degeneration
        self.d_AL_form_man += -self.tau*1*df.div(self.V)/df.CellVolume(mesh)**2*df.dx(domain=mesh)
        
        # constraint (element size derivative)
        self.d_AL_form_man += divE(self.tE, self.V)*0.5*self.rho[0]*(self.d[0]('+') - self.logterm - self.b[0]('+'))**2*df.dS(domain=mesh) 
        self.d_AL_form_man += -divE(self.tE, self.V)*0.5*self.rho[0]*self.b[0]('+')**2*df.dS(domain=mesh)
        # constraint (material derivative) 
        self.d_AL_form_man += self.rho[0]*np.inner(self.d[0]('+') - self.logterm - self.b[0]('+'), -self.dlogterm(self.V) )*df.dS
        
        
        self.d_AL_form_auto = df.derivative(self.AL_form_auto, self.x, self.V)
        self.H_AL_form_auto = df.derivative(self.d_AL_form_auto, self.x, self.W)

        self.d_AL_form = self.d_AL_form_man + self.d_AL_form_auto
        
        
        # derivatives for circumcenters
        self.d_AL_form_verts = df.derivative(self.AL_form, self.verts, df.TestFunction(self.CCSpace))
        self.H_AL_form_verts = df.derivative(self.d_AL_form_verts, self.verts, df.TrialFunction(self.CCSpace))
        
        # compute mixed hessians with respect to CC and shap
        self.H_AL_form_verts_mixed_man = df.derivative(self.d_AL_form_man, self.verts, df.TrialFunction(self.CCSpace))
        self.H_AL_form_verts_mixed_auto = df.derivative(df.derivative(self.AL_form_auto, self.verts, df.TrialFunction(self.CCSpace)), self.x, df.TestFunction(self.CG))
        self.H_AL_form_verts_mixed = self.H_AL_form_verts_mixed_man + self.H_AL_form_verts_mixed_auto
              
        # derivatives with respect to u
        self.d_AL_form_aux = df.derivative(self.AL_form, self.u, df.TestFunction(self.RT))
        self.H_AL_form_aux  = df.derivative(self.d_AL_form_aux, self.u, df.TrialFunction(self.RT))
        
        # compute mixed hessians with respect to cc 
        self.H_AL_form_aux_mixed_man = df.derivative(self.d_AL_form_man, self.u, self.w)
        self.H_AL_form_aux_mixed_auto = df.derivative(df.derivative(self.AL_form_auto, self.u, self.w), self.x, df.TestFunction(self.CG))
        self.H_AL_form_aux_mixed = self.H_AL_form_aux_mixed_man + self.H_AL_form_aux_mixed_auto 
        # transposed
        self.H_AL_form_aux_mixedT_man = df.derivative(ufl.replace(self.d_AL_form_man, {self.V : self.W}), self.u, self.v)
        self.H_AL_form_aux_mixedT_auto = df.derivative(df.derivative(self.AL_form_auto, self.u, self.v), self.x, df.TrialFunction(self.CG))
        self.H_AL_form_aux_mixedT = self.H_AL_form_aux_mixedT_man + self.H_AL_form_aux_mixedT_auto 
        
        
        
        # second derivative of AL
        # fidelity
        self.H_AL_form_man = df.inner(self.W, self.V)*df.dP
        # term to prevent mesh degeneration        
        self.H_AL_form_man += -self.tau*1 / df.CellVolume(mesh)**2 * Hsurf(self.n, self.V, self.W) * df.dx(domain=mesh)
        self.H_AL_form_man += 2*self.tau*1 / df.CellVolume(mesh)**2 * df.div(self.V) * df.div(self.W) * df.dx(domain=mesh)
        
        # constraint (element size second derivative)
        self.H_AL_form_man += Hedge(self.tE,self.V,self.W)*0.5*self.rho[0]*(self.d[0]('+') - self.logterm - self.b[0]('+'))**2*df.dS(domain=mesh) 
        self.H_AL_form_man += -Hedge(self.tE,self.V,self.W)*0.5*self.rho[0]*self.b[0]('+')**2*df.dS(domain=mesh)
        # pt ist norm erhaltend
        
        # constraint (element size and material mixed second derivative)
        self.H_AL_form_man += divE(self.tE, self.W)*self.rho[0]*np.inner(self.d[0]('+') - self.logterm - self.b[0]('+'), -self.dlogterm(self.V) )*df.dS
        self.H_AL_form_man += divE(self.tE, self.V)*self.rho[0]*np.inner(self.d[0]('+') - self.logterm - self.b[0]('+'), -self.dlogterm(self.W) )*df.dS
        
 
        # constraint  (second material derivative)
        self.H_AL_form_man += self.rho[0]*np.inner(-self.dlogterm(self.V),-self.dlogterm(self.W) )*df.dS 
        self.H_AL_form_man += self.rho[0]*np.inner(self.d[0]('+') - self.logterm - self.b[0]('+'), -self.ddlogterm )*df.dS
        
           
        #self.H_AL_form = self.H_AL_form_auto + self.H_AL_form_man + self.H_dofform
        self.H_AL_form = self.H_AL_form_man + self.H_dofform + self.H_AL_form_auto 
        
        # mixed CC and aux hessian
        self.H_AL_form_verts_aux_mixed_V = df.derivative(df.derivative(self.AL_form, self.verts, df.TestFunction(self.CCSpace)), self.u, self.w)
        self.H_AL_form_verts_aux_mixed_U = df.derivative(df.derivative(self.AL_form, self.verts, df.TrialFunction(self.CCSpace)), self.u, self.v)
        

        
        
        self.history = []
        # no fixed vertices
        self.fixed_BC = None
        #assembled hessians
        self.assembled_hesses = None
        self.assembled_hesses_aux = None
        self.d_AL_aux_assembled.vector().set_local( df.assemble(self.d_AL_form_aux))
        
        
        # allocate memory for the assembled matrices for the u problem
        self.hess1_aux = df.assemble(self.dofmapform)
        self.hess2_aux = df.assemble(self.H_AL_form_aux)
        self.hess3_aux = df.assemble(self.dofmapformT)
        
        
        # allocate memory for the assembled matrices fior the shape problem
        self.hess0_shape = df.assemble(self.H_AL_form)
        self.hess1_shape = df.assemble(self.H_AL_form_verts_mixed)
        self.hess2_shape = df.assemble(self.H_AL_form_verts)
        self.hess3_shape = df.assemble(self.d_dofformT)
        self.hess4_shape = df.assemble(self.d_dofform)
        self.hess5_shape = df.assemble(self.H_AL_form_aux_mixed)
        self.hess6_shape = df.assemble(self.H_AL_form_aux)
        self.hess7_shape = df.assemble(self.H_AL_form_verts_aux_mixed_U)
        
        
        
        #residuals
        self.primal_res_form0 = (self.d[0]('+') - self.logterm )**2*df.dS
        self.primal_res_form1 = ( self.d[1] - self.grad(self.u) )**2*df.dx
        self.primal_res_form2 = ( self.d[2]('+') - tan_jump(self.u, self.tE, self.mu))**2*(self.orth1('+')+self.orth2('+'))/2*df.dS
        
        self.log_old = df.Function(self.skeletonDG)
        self.grad_old = df.Function(self.DGten)
        self.jump_old = df.Function(self.skeletonDGveclin)
        
        self.log_old.vector().set_local(df.assemble( df.inner(df.TestFunction(self.skeletonDG)('+'), self.logterm)/df.FacetArea(self.mesh)('+')*df.dS).get_local())
        self.grad_old.vector().set_local(df.assemble( df.inner(df.TestFunction(self.DGten),  self.grad(self.u) )/df.CellVolume(self.mesh)*df.dx).get_local())
        self.jump_old.vector().set_local(df.assemble( df.inner( df.TestFunction(self.skeletonDGveclin)('+'), tan_jump(self.u, self.tE, self.mu))*(self.orth1('+')+self.orth2('+'))/df.FacetArea(self.mesh)('+')*df.dS).get_local())
             
        self.dual_res_form0 = (self.rho[0]*(self.log_old('+') - self.logterm ))**2*df.dS
        self.dual_res_form1 = (self.rho[1]*(parallelTransportTensor(self.grad_old, self.n_0, self.n) - self.grad(self.u) ))**2*df.dx
        self.dual_res_form2 = (self.rho[2]*(parallelTransport(self.jump_old('+'), self.n_0('+'), self.n('+')) - tan_jump(self.u, self.tE, self.mu)))**2*(self.orth1('+')+self.orth2('+'))*df.dS
        
        # list for scalar quantities to be filled by AL.log
        self.history = []

    # evaluate AL    
    def val(self, vec=None):
        # if vec for mesh coordinates is provided, update CCs and u using the tangent RT dofs
        if not vec is None:
            self.set_mesh_coords(vec)
            self.set_cc_vecs(vec)
            self.u.vector().set_local( df.assemble(self.dofform))       
        return  df.assemble(self.AL_form)
      
    #evaluate derivative of AL
    def derivative(self, vec=None):        
        if not vec is None:
            self.set_mesh_coords(vec)
            self.set_cc_vecs(vec)
            self.u.vector().set_local( df.assemble(self.dofform))    
            
        # try to fix mesh degenerations due to early gradient steps
        tmp = df.assemble( df.inner(self.n('+'), self.n('-'))*df.TestFunction(self.skeletonDG)('+')/df.FacetArea(self.mesh)('+')*df.dS).get_local()
        k = 0
        while np.min(tmp) < -np.sqrt(0.5):
            k+=1
            #print("mesh degenerated (some normals are almost antipodal), trying to fix the situation")
            utils.utils.repairEdges(self.mesh,self.n)
            self.set_cc_vecs(self.get_mesh_coords())
            self.u.vector().set_local( df.assemble(self.dofform))
            
            tmp = df.assemble( df.inner(self.n('+'), self.n('-'))*df.TestFunction(self.skeletonDG)('+')/df.FacetArea(self.mesh)('+')*df.dS).get_local()
            if k >10:
                break
        
        # compute shape derivative
        vec =  df.assemble(self.d_AL_form) 
        # add chainrule for CCs
        vec.add_local( df.assemble(self.d_AL_form_verts).get_local()@self.CCmat)
        # add chainrule for DRT function
        self.d_AL_aux_assembled.vector().set_local( df.assemble(self.d_AL_form_aux))
        vec.add_local( df.assemble(self.d_dofform_aux_assembled))
        
        return vec
    

    #evaluate Hess of AL
    def Hess(self, d = None, vec=None):
        if not vec is None:
            self.set_mesh_coords(vec)
            self.set_cc_vecs(vec)
            self.u.vector().set_local( df.assemble(self.dofform))
            
        if self.assembled_hesses is None:
            #assemble all the hessians
            starttime = time.time()
            self.d_AL_aux_assembled.vector().set_local( df.assemble(self.d_AL_form_aux))
            df.assemble(self.H_AL_form, tensor = self.hess0_shape)
            if self.with_aux:
                df.assemble(self.H_AL_form_verts_mixed, tensor = self.hess1_shape)
                df.assemble(self.H_AL_form_verts, tensor = self.hess2_shape)
                df.assemble(self.d_dofform, tensor = self.hess4_shape)
                df.assemble(self.H_AL_form_aux_mixed, tensor = self.hess5_shape)
                df.assemble(self.H_AL_form_aux, tensor = self.hess6_shape)
                df.assemble(self.H_AL_form_verts_aux_mixed_U, tensor = self.hess7_shape)

            
            self.assembled_hesses = [self.hess0_shape, self.hess1_shape, self.hess2_shape, self.hess4_shape, self.hess5_shape, self.hess6_shape, self.hess7_shape]#,self.hess1_aux,self.hess3_aux, self.hess8_shape, self.hess9_shape] 
            print("assemble took", time.time() - starttime)
        if d is not None:
        
            #from IPython import embed; embed()
            #starttime = time.time()
            
            # shape shape
            vec =  self.assembled_hesses[0]*d
            
            if self.with_aux:
                #verts shape 
                self.assembled_hesses[1].transpmult(d, self.dverts.vector())
                vec.add_local( self.CCmat.T@self.dverts.vector().get_local() )
                
                # shape verts
                self.dverts.vector().set_local( self.CCmat@d.get_local())
                vec.add_local( (self.assembled_hesses[1]*self.dverts.vector()).get_local())
                # verts verts 
                vec.add_local( self.CCmat.T@(self.assembled_hesses[2]*self.dverts.vector()).get_local())
                 
                # shape u 
                self.assembled_hesses[3].transpmult(d, self.du.vector())
                vec.add_local( (self.assembled_hesses[4]*self.du.vector()).get_local())
                
                # verts u 
                self.assembled_hesses[6].transpmult(self.du.vector(), self.vertstmp.vector())
                vec.add_local( self.CCmat.T@self.vertstmp.vector().get_local() )
                
                
                # u shape 
                self.assembled_hesses[4].transpmult(d, self.utmp.vector())
                # finish u shape , u u, u, verts 
                tmpvec = (self.utmp.vector()+self.assembled_hesses[5]*self.du.vector() + self.assembled_hesses[6]*self.dverts.vector())
                vec.add_local( (self.assembled_hesses[3]*tmpvec).get_local())

            return vec

        return None 



          

    # move mesh to input vec
    def set_mesh_coords(self,vec):
        # reset assembled hessians
        self.assembled_hesses = None
        self.assembled_hesses_aux = None
        # set mesh coordinates to vec by moving it by vec - current_cords
        movevec = vec - df.assemble( df.inner(self.x, self.V)*df.dP )
        self.movefun.vector().set_local(movevec)
        df.ALE.move(self.mesh, self.movefun)
        

        
    def get_mesh_coords(self):
        return df.assemble( df.inner(self.x, self.V)*df.dP )

    # sets the vertices A,B,C,M used to computed the circumcenters and h_E
    def set_cc_vecs(self, vec):
        self.verts.vector().set_local(self.CCmat@vec)

    def update_nonsmooth(self):

        vec0 = df.assemble(df.inner(df.TestFunction(self.skeletonDG)('+'), self.logterm)/df.FacetArea(self.mesh)('+')*df.dS).get_local() + self.b[0].vector().get_local()
        normvec0 = np.repeat(np.linalg.norm(vec0.reshape(-1,1), axis = 1), 1)
        self.d[0].vector().set_local(vec0/np.maximum(1e-13,normvec0)*np.maximum(0, normvec0 - float(self.alpha[1])/float(self.rho[0])  ))
        
        if self.with_aux:
            vec1 = df.assemble(df.inner(df.TestFunction(self.DGten), self.grad(self.u) )/df.CellVolume(self.mesh)*df.dx).get_local() + self.b[1].vector().get_local()
            normvec1 = np.repeat(np.linalg.norm(vec1.reshape(-1,27), axis = 1), 27)
            self.d[1].vector().set_local(vec1/np.maximum(1e-13,normvec1)*np.maximum(0, normvec1 - float(self.alpha[0])/float(self.rho[1])))
            
            vec2 = df.assemble( df.inner( df.TestFunction(self.skeletonDGveclin)('+'), tan_jump(self.u, self.tE, self.mu))*(self.orth1('+')+self.orth2('+'))/df.FacetArea(self.mesh)('+')*df.dS).get_local() + self.b[2].vector().get_local()
            normvec2 = np.repeat(np.linalg.norm(vec2.reshape(-1,3), axis = 1), 3)
            self.d[2].vector().set_local(vec2/np.maximum(1e-13,normvec2)*np.maximum(0, normvec2 - float(self.alpha[0])/float(self.rho[2])))
       
        
    #@profile
    def update_aux(self):
        
        # assemble matrixces
        df.assemble(self.dofmapform, tensor = self.hess1_aux)
        df.assemble(self.H_AL_form_aux, tensor = self.hess2_aux)
        df.assemble(self.dofmapformT, tensor = self.hess3_aux)
        
       
        # compute gradient
        grad = self.hess1_aux*df.assemble(self.d_AL_form_aux) 
        
        # build hessian matrix composed via chainrules
        customHess = CustomMatrix(df.as_backend_type(self.hess1_aux).mat(), df.as_backend_type(self.hess2_aux).mat(), df.as_backend_type(self.hess3_aux).mat(), df.as_backend_type(self.utmp1.vector()).vec(), df.as_backend_type(self.utmp2.vector()).vec())
        A = PETSc.Mat().createPython([self.hess1_aux.size(0),self.hess3_aux.size(1)], comm=df.MPI.comm_world)
        A.setPythonContext(customHess)
        A.setUp()   
        APetsc = df.PETScMatrix(A)
        
        solver = df.PETScKrylovSolver("cg", "none")
        solver.parameters["absolute_tolerance"] = 1e-5
        solver.parameters["error_on_nonconvergence"] = False
        c = solver.solve(APetsc,self.dudof.vector(),-grad)
        print( "solver exit code was ", c)

        
        self.udof.vector().add_local(self.dudof.vector())
        self.u.vector().set_local(self.hess3_aux*self.udof.vector())
        
        return
        
        
        

    def update_multipliers(self):
        # fix mesh degeneration if it occured
        tmp = df.assemble( df.inner(self.n('+'), self.n('-'))*df.TestFunction(self.skeletonDG)('+')/df.FacetArea(self.mesh)('+')*df.dS).get_local()
        k = 0
        while np.min(tmp) < -np.sqrt(0.5):
            k+=1
            #print("mesh degenerated (some normals are almost antipodal), trying to fix the situation")
            utils.utils.repairEdges(self.mesh,self.n)
            tmp = df.assemble( df.inner(self.n('+'), self.n('-'))*df.TestFunction(self.skeletonDG)('+')/df.FacetArea(self.mesh)('+')*df.dS).get_local()
            if k >10:
                break
                
                
        #actual multiplier updates       
        vec0 =  df.assemble(df.inner(df.TestFunction(self.skeletonDG)('+'), self.b[0]('+') + self.logterm - self.d[0]('+'))/df.FacetArea(self.mesh)('+')*df.dS).get_local() 
        self.b[0].vector().set_local(vec0)
        if self.with_aux:
            vec1 = df.assemble(df.inner(df.TestFunction(self.DGten), parallelTransportTensor(self.b[1],self.n_0, self.n) +  self.grad(self.u) - parallelTransportTensor(self.d[1],self.n_0, self.n) )/df.CellVolume(self.mesh)*df.dx).get_local()
            self.b[1].vector().set_local(vec1)
        
            vec2 = df.assemble( df.inner( df.TestFunction(self.skeletonDGveclin)('+'), parallelTransport(self.b[2], self.n_0, self.n)('+') +  tan_jump(self.u, self.tE, self.mu) - parallelTransport(self.d[2], self.n_0, self.n)('+'))*(self.orth1('+')+self.orth2('+'))/df.FacetArea(self.mesh)('+')*df.dS).get_local()
            self.b[2].vector().set_local(vec2)
        
        
        
        
        
    def update_tracking(self):
        pass
        #self.tracking_data = df.project(self.x, self.CG) 
        
        
    def update_penalty_param(self, factor = 5, scaling = 1.2):
        # update the penalty parameter
        if self.adapt_penalty:
            #print( "auxres", self.aux_residual())
            primal_res = self.primal_residuals()
            dual_res = self.nonsmooth_residuals()
            
            if self.with_aux:
                N = 3
                aux_res = self.aux_residual()
            else:
                N = 1
                aux_res = 0.
            
            for i in range(N):
                if primal_res[i] >= factor*dual_res[i] and sum(primal_res) >= factor*aux_res:
                    # rho = rho*scaling
                    if i == 0 and float(self.rho[0]) <= 1000*float(self.alpha[1]):
                        self.rho[i].assign(self.rho[i]*scaling)
                        self.b[i].assign(self.b[i]/scaling)
                    if i> 0 and float(self.rho[i]) <= 1000*float(self.alpha[0]):
                        self.rho[i].assign(self.rho[i]*scaling)
                        self.b[i].assign(self.b[i]/scaling)
                    
                    print("new penalty parameter+: ", i, float(self.rho[i]))
                if dual_res[i] >= factor*primal_res[i] or aux_res >= factor*sum(primal_res):
                    # rho = rho/scaling
                    self.rho[i].assign(self.rho[i]/scaling)
                    self.b[i].assign(self.b[i]*scaling)
                    print("new penalty parameter-: ", i, float(self.rho[i]))
        
    # save log of this iteration
    def finish_iteration(self):
        # save old values for dual residuals
        self.log_old.vector().set_local(df.assemble( df.inner(df.TestFunction(self.skeletonDG)('+'), self.logterm)/df.FacetArea(self.mesh)('+')*df.dS).get_local())
        self.grad_old.vector().set_local(df.assemble( df.inner(df.TestFunction(self.DGten),  self.grad(self.u) )/df.CellVolume(self.mesh)*df.dx).get_local())
        self.jump_old.vector().set_local(df.assemble( df.inner( df.TestFunction(self.skeletonDGveclin)('+'), tan_jump(self.u, self.tE, self.mu))*(self.orth1('+')+self.orth2('+'))/df.FacetArea(self.mesh)('+')*df.dS).get_local())
        # save old normal for parallel transports
        self.n_0.assign(df.project(self.n, self.DGvec))
        
        
        print( df.norm(self.d[0].vector()), df.norm(self.d[1].vector()),df.norm(self.d[2].vector()))
    
    def primal_residuals(self):
        return [np.sqrt(df.assemble(self.primal_res_form0)),np.sqrt(df.assemble(self.primal_res_form1)),np.sqrt(df.assemble(self.primal_res_form2))]

        
    def nonsmooth_residuals(self):
        return [np.sqrt(df.assemble(self.dual_res_form0)),np.sqrt(df.assemble(self.dual_res_form1)),np.sqrt(df.assemble(self.dual_res_form2))]

    def aux_residual(self):
        df.assemble(self.dofmapform, tensor = self.hess1_aux)
        grad = self.hess1_aux*df.assemble(self.d_L_form_aux) 
        edgelen = df.assemble( df.TestFunction(self.skeletonDG)('+')*df.dS(domain=self.mesh)).get_local()
        return np.linalg.norm(grad.get_local()/np.repeat(edgelen,2))


    def log(self, infos):
        pres = self.primal_residuals()
        nsres = self.nonsmooth_residuals()
        auxres = self.aux_residual()
        combres = np.sqrt(sum( [np.linalg.norm(v)**2 for v in pres+nsres+[auxres]]))
        current_state = [ combres, *pres, *nsres, auxres, float(self.rho[0]),float(self.rho[1]),float(self.rho[2]), time.time(), *infos]
        self.history += [current_state]
        
        
    def grad(self, u):
        g = df.grad(u)
        i, j, k, = ufl.indices(3)
        return df.as_tensor(0.5*g[i,j,k]+0.5*g[i,k,j], (i,j,k))
        
class CustomMatrix():
    
    def __init__(self, mat1, mat2, mat3, tmpvec1, tmpvec2):
        self.mat1 = mat1
        self.mat2 = mat2
        self.mat3 = mat3
        self.tmpvec1 = tmpvec1
        self.tmpvec2 = tmpvec2
        pass
    
    
    def mult(self, mat, x,y):
        self.mat3.mult(x,self.tmpvec1)
        self.mat2.mult(self.tmpvec1,self.tmpvec2)
        self.mat1.mult(self.tmpvec2,y)
        
