from dolfin import *
set_log_level(30)

import numpy as np
import time
#@profile
def ADMM(AL, smooth_steps, cb = None, max_outer_iter=1000, max_inner_iter=10, stepsize = 1e-4, verbose=False, with_aux=True):
    #AL.log([max_inner_iter, 0])
    if with_aux == False:
        AL.with_aux = False
    for i in range(max_outer_iter): 
        if verbose:
            print("i outer = ", i)
        if cb != None:
            cb(AL, i)        
        
        AL.update_nonsmooth()    
       
        if with_aux:
            AL.update_aux()
          
        stepsize, inner_iterations = smooth_steps(AL, max_iter=max_inner_iter, stepsize=stepsize, verbose=verbose)
        
        AL.update_multipliers()
        
        if i % 10 == 0:
            AL.log([inner_iterations, i+1])
               
        print("primal residuals", AL.primal_residuals())
        print("dual residuals", AL.nonsmooth_residuals())
        
        if i > -1 and stepsize > 0: 
            AL.update_penalty_param()
        
        AL.finish_iteration()
        

            
    
    
    history_array = np.array(AL.history)
    history_array[:, 11] -= history_array[0,11]
    time_per_iteration = np.hstack([np.nan, history_array[1:,11]  - history_array[:-1, 11]])
    history_array = np.hstack( [history_array[:, :12], time_per_iteration[:, np.newaxis], history_array[:, 12:]])
    
    np.savetxt("history.txt", history_array)
