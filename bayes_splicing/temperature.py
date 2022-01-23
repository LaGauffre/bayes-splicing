# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:21:51 2020

@author: pierr
"""

import numpy as np 
import numba as nb

@nb.jit(nopython=True)
def temperature_search(particles, log_probs, target, γ_prev, err):
    """
    Provides the next temperature given the previous generation of particles.

    Parameters
    ----------
    particles : ndarray 
        cloud of particles.
    target : int
        Effective sample size target, default is popSize / 2.
    γ_prev : float
        Previous temperature.
    err: float
        Temperature threshold
        
    Returns
    -------
    list
    A list that provides the next temperature, the unnormalized weights w, 
    the normalized weights W, and the effective sample size.
    
    Example
    -------
    k, α, θ = 1/2, 1/2, 5 
    X = sim_wei_par(100, k, α, θ)
    popSize, a, b, c, ρ, n_step_max,loss_model, prior, parms_names = 20, [0,0,0], [10,10,10],\
    0.99, 1/2, 25,"wei_par", "uniform", ['k','α', 'θ']
    init_cloud = sim_uniform_prior(a, b, parms_names, popSize)
    init_cloud['w'] = np.ones(popSize)
    init_cloud['W'] = np.ones(popSize) / popSize

    particles, target, γ_prev, model, method = init_cloud[parms_names] ,  
    ρ * popSize, 0, "wei-par", "likelihood annealing"
    temperature_search(X, particles, popSize, target, γ_prev, model, method)
    """
    γ_up, γ_down = 2, γ_prev
    popSize, d = particles.shape
    
    if np.any(np.isnan(log_probs)):
        log_probs[np.where(np.isnan(log_probs))] = -np.inf
    while γ_up - γ_down > err:
        γ_new = (γ_up + γ_down) / 2
        
        logw =  log_probs*(γ_new - γ_prev)
        if np.all(logw == - np.inf):
            #  This will cancel out this iteration 
            # in the log marginal likelihood computation
            logw = np.log(np.ones(popSize))
            W = 1 / popSize *np.ones(popSize)
            ESS = 0
        else:
            W = np.exp(logw - max(logw)) / np.sum(np.exp(logw - max(logw)))
            ESS =  1 / np.sum(W**2)
        

        if ESS == target:
            break
        else:
            if ESS < target:
                γ_up = γ_new
            else:
                γ_down = γ_new
                if γ_down > 1:
                    break
    if γ_new >= 1: 
        γ_new = 1
        logw =  log_probs*(γ_new - γ_prev)
         
        W = np.exp(logw - max(logw)) / np.sum(np.exp(logw - max(logw)))

        ESS =  1 / np.sum(W**2)
    return(γ_new, logw, W, int(ESS))  



# def batch_size_search(particles, target, n_prev, X, loss_model):
#     """
#     Provides the next sample size to be considered given the 
#     previous generation of particles.

#     Parameters
#     ----------
#     particles : ndarray 
#         cloud of particles.
#     target : int
#         Effective sample size target, default is popSize / 2.
#     n_prev : float
#         Previous sample size.
        
#     Returns
#     -------
#     list
#     A list that provides the next sample size, the unnormalized weights w, 
#     the normalized weights W, and the effective sample size.
#     """
#     popSize, d = particles.shape
    
#     logd = logd_wrap(particles, loss_model)
    
#     logws = np.cumsum(np.array([logd(x) for x in X[n_prev:]]), axis = 0)
#     selec = np.where(np.all(logws,axis = 1) > -np.inf )[0]
    
#     if selec.size > 0:
       
#         Ws = np.array([
#             np.exp((logws[k,:] - max(logws[k,:]))) / sum(np.exp(logws[k,:] - max(logws[k,:]))) 
#             for k in selec])
#         ESSs = 1/np.sum(Ws**2, axis = 1)
#         Δ_ESS = np.abs(ESSs-target)
#         k = np.where( Δ_ESS == np.nanmin(Δ_ESS))[0][0]
#         n_new, logw, W, ESS = n_prev + (k+1), logws[k,:], Ws[k,:],int(ESSs[k])
#     else:
#         logw = np.log(np.ones(popSize))
#         W = 1 / popSize * np.ones(popSize)
#         ESS = 0
#         n_new = n_prev +1
    
#     return(n_new, logw, W, int(ESS))  

