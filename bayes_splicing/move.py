# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:21:38 2020

@author: pierr
"""

import numpy as np
import numba as nb
# @nb.jit(nopython=True)
# n_moves, step_size, log_prob, log_prob_prior, init_parms, τ, d = 1, np.diag(cloud_cov), log_prob, \
#                                log_prob_prior, particles_resampled[0], τ, d
@nb.jit(nopython=True)
def Gibbs_move(n_moves, step_size, log_prob, log_prob_prior, init_parms, τ, d):   
    """
    Function to perform Gibbs moves.

    Parameters
    ----------
    n_moves : int 
        number of moves.
    step_size: Array
        size of the steps in the MH within Gibbs move
    log_prob: function
        log likelihood function
    log_prob_prior: function
        log probability of the prior distribution
    init_parms: array
        initial value of the parameters
    γ: float
        Temperature
    d: int
        number of parameters
        
    Returns
    -------
    dataframe
    trace of each parameter and acceptance rate.
    
    Example
    -------
    k, α, θ = 1/2, 1/2, 5 
    X = sim_wei_par(100, k, α, θ)
    n_moves, step_size, log_prob, log_prob_prior, init_parms, γ, parms_names = 2, [0.05, 0.01, 2], \
    logp_wei_par(X), uniform_prior(np.array([0,0,0]), np.array([10,10,10])), np.array([1,1,1]), 1, ['k', 'α', 'θ']
    trace, acceptance = Gibbs_move(n_moves, step_size, log_prob, log_prob_prior, init_parms, γ, parms_names)
    """
    accepted_matrix = [np.repeat(False,d)]
   
    trace = [init_parms]
    

    # epsilon = np.random.multivariate_normal(mean = np.zeros(d), 
    #                                       cov = np.diag(step_size), size = n_moves)
    
    # for noise in np.atleast_2d(epsilon):
    for j in range(n_moves):
        parms_perturbed = trace[-1].copy()
        parms_new = trace[-1].copy()
        accepted = []
        for i in range(d):
            noise = np.random.normal(0, np.sqrt(step_size[i]))
            parms_perturbed[i] = parms_perturbed[i] + noise
            
            parms_new[i] = parms_perturbed[i]
            # acceptance rate
            old_log_p = τ * log_prob(trace[-1]) + log_prob_prior(trace[-1])
            new_log_p = τ * log_prob(parms_new) + log_prob_prior(parms_new)        
            ###
            acc =  new_log_p - old_log_p 
            ###       
            if np.log(np.random.rand()) < acc:
                parms_new[i] = parms_perturbed[i]
                accepted.append(True)
            else:
                parms_new[i] = trace[-1][i]
                # np.put(parms_new, i, trace[-1][i])
                accepted.append(False)
        trace.append(parms_new) 
        accepted_matrix.append(np.array(accepted))
                
     
    return(trace, accepted_matrix)






