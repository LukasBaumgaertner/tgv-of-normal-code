from truncatedCG import truncatedCG
import dolfin as df
df.set_log_level(30)
from petsc4py import PETSc
import numpy as np


#Algorithm 3´
def Newton(AL, max_iter=100, stepsize=1e-4, tol_smooth=1e-5, verbose = False):

    for i in range(max_iter):
        #define preconditioner solver from icc of H1 inner product
        
        pc_form = df.inner(AL.V, AL.W)*df.dx + 1e-2*df.inner(df.grad(AL.V), df.grad(AL.W))*df.dx
        #pc_form = [df.inner(AL.V,AL.W)*df.dP, df.inner(AL.vdof, AL.wdof)('+')/df.FacetArea(AL.mesh)('+')**2*df.dS]
        pc = define_preconditioner(pc_form)
        M_solve = lambda x, y: pc(x, y)
    
        # compute gradient wrt M inner product, slope in its direction as well as its norm. Apply BC to derivative and gradient if there are fixed vertices. 
        der_AL = AL.derivative()
            
        # create vector of same shape as der_AL and solve the preconditioned derivative into that
        grad_AL = der_AL.copy()
        #from IPython import embed; embed()
        M_solve(grad_AL, der_AL)
        
            
        slope_grad = grad_AL.inner(der_AL)
        norm_grad = np.sqrt(slope_grad)
        
        if verbose:
            print("i inner = ", i, "resnorm", norm_grad, "stepsize ", stepsize)
            
        # if tol is reached stop     
        if norm_grad < tol_smooth:
            return stepsize, i
        
    
        # get hessian matrix and its action as a function as well as the gradient
        #Hess_AL = AL.Hess()
        Hess_action = lambda d: AL.Hess(d=d)
           
        # get descent direction
        d, flag = truncatedCG(-der_AL, Hess_action, M_solve, bc=AL.fixed_BC)

        # directional derivative in direction d
        slope_d = der_AL.inner(d) 
        # starting point for linesearch
        x = AL.get_mesh_coords()
        
        # if CG didnt do a single step (flag = False) or d is a very bad descent direction fall back to gradient and gradient LS
        fun0 = df.Function(AL.CG)
        fun0.vector().set_local(d)
        
        
        norm_d = np.sqrt(df.assemble( (pc_form*fun0)*fun0))
        if flag and slope_d <= -min(0.1, 1e-6*norm_d**0.1)*norm_d*norm_grad:
            #newton step
            stepsize = armijo_move(x,AL.val, d, slope_d, 0.5, 0.4, 1)
        else:
            #use gradient
            if verbose:
                print("falling back to gradient step")
            stepsize = armijo_move(x,AL.val, -grad_AL, -slope_grad, 0.5, 0.7, stepsize*2)
            
    
    
    if verbose:
        der_AL = AL.derivative()
        if AL.fixed_BC != None:
            AL.fixed_BC.apply(der_AL)
            
        grad_AL = der_AL.copy()
        M_solve(grad_AL, der_AL)
        
        if AL.fixed_BC != None:
            AL.fixed_BC.apply(grad_AL)
        slope_grad = grad_AL.inner(der_AL)
        norm_grad = np.sqrt(slope_grad)
        
        print("i inner = ", i, "resnorm", norm_grad, "stepsize ", stepsize)
        
            
    return stepsize, i+1
    
    
    
    
def GradientDescent(AL, max_iter=100, stepsize=1e-2, tol_smooth = 1e-3, verbose = False):
    
    der_AL = AL.derivative()
    
    for i in range(max_iter):
            
        
        der_AL = AL.derivative()     

        #define inner product as H1 inner product
        Inner = df.inner(AL.V, AL.W)*df.dx + 1e-2*df.inner(df.grad(AL.V), df.grad(AL.W))*df.dx
        
        # compute gradient wrt H1 inner product 
        d = -der_AL.copy()
        df.solve(df.assemble(Inner), d, -der_AL)
        #df.solve(df.assemble(Inner[1]), d[1], -der_AL[1])
        slope = d.inner(der_AL)
        norm_der = np.sqrt(-slope)
        if verbose:
            print("i inner = ", i, "resnorm", norm_der, "stepsize ", stepsize)
        if norm_der < tol_smooth:
            return stepsize, i+1
    
    
        # get staring  point for linesearch     
        
        x = AL.get_mesh_coords()
        #linesearch
        stepsize = armijo_move(x, AL.val, d, slope, 0.5, 0.2, stepsize*2)
    

    return stepsize, i
    
    
def armijo_move(x, f, d, slope, tau, c, astart):
    
    a = astart
    
    t = -c*slope

    f_x = f(x)   
    # function evaluations move the mesh
    f_moved = f( x+a*d)
    
    while f_x - f_moved < a*t and a > 1e-12:
        a = a*tau
        f_moved = f( x+a*d)

    return a




def define_preconditioner(form, type="icc"):
    preconditionerMat = df.assemble(form)
    ksp = PETSc.KSP().create()
    ksp.setOperators(df.as_backend_type(preconditionerMat).mat())
    ksp.setType('preonly')
    ksp.setConvergenceHistory()
    ksp.getPC().setType(type)
    solver = df.PETScKrylovSolver(ksp)
    solver.set_reuse_preconditioner(True) 
    return solver.solve
