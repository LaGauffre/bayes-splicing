# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:21:21 2020

@author: pierr
"""

import pandas as pd
import numpy as np
import scipy.special as sp
from joblib import Parallel, delayed
from .loss_distribution import *
from .prior_distribution import *
from .independent_priors import *
from .temperature import temperature_search
from .move import Gibbs_move

def smc(X, f, popSize, prior_spliced_model, ρ = 0.5,
                             c = 0.99, n_step_max = 25, err = 1e-6, paralell = False, n_proc = 4, verbose = True):
    """
    Sequential Monte Carlo Sampler of the posterior distribution.

    Parameters
    ----------
    X: array
        loss data required to evaluate the likelihood
    loss_model: string
        loss model being fitted
    parms_names: array
        names of the loss model parameters, 
        first the shape parameter of the belly distribution
        second the tail index
        third the threshold between small and large claims
    popSize: int
        number of particles sampled, size of the cloud
    model_prior: string
        prior distribution
    a, b: arrays 
        prior distribution parameters
    ρ: float
        tuning parameter for the target effective sample size 
    move_type: string 
        type of moves to choose in ("Metropolis", "Gibbs", "Indpendent)"
    c : float
        Calibrate the number of steps required so that the probability that 
        each particle is moved at least once equals c.
    n_step_max: int
        limits the number of steps
    err: float
        Temperature threshold
    verbose: boolean
        Whether to print the steps
    
    Returns
    -------
    list
    A list that provides the posterior sample along with the smc estimator of 
    the marginal likelihood.
    
    Example
    -------

    """
    f.set_logp(X), prior_spliced_model.set_logp()
    log_prob_prior, log_prob, d = prior_spliced_model.logp, f.logp, f.d
    
    # Generation counter
    g = 0
    if verbose:
        print('Sample generation ' + str(g) + " from the " + prior_spliced_model.name +
          " prior distribution")
    # Initialisation of the particle cloud
    init_cloud = prior_spliced_model.sample(popSize)
    # init_cloud = sim_prior_wrap(model_prior, a, b, parms_names, popSize)
    init_cloud['logw'] = np.log(np.ones(popSize))
    init_cloud['W'] = np.ones(popSize) / popSize
    # This initial particle cloud is placed inside a list
    clouds = []
    clouds.append(init_cloud)
    
    # Temperature sequence either true temperature or proportion of observations
    τ_seq = np.array([0])

   
    # We keep on iterating until the temperature reaches 1
    while τ_seq[-1] < 1:
        
        g = g + 1
        particles = clouds[g-1][f.parm_names].values
        log_probs = np.array([log_prob(particle) for particle in particles])
        
        # Updating temperature sequence 
        τ, logw, W, ESS = temperature_search(particles,
                                             log_probs,ρ * popSize, τ_seq[-1],
                                             err)
        
       
        τ_seq = np.append(τ_seq, τ)
       
        step_sizes = np.array([np.sum((particles[:,k]  - np.sum(particles[:,k] * W))**2 * W) 
                  for k in range(d)]) * 2.38 / np.sqrt(d) / (1 - np.sum(W**2))
    
        particles_resampled = particles[np.random.choice(popSize,popSize, p = W)]
        
        def move_particle_trial(particle):
            
            trace, acceptance = Gibbs_move(1, 
                                            step_sizes, 
                                            log_prob, 
                                            log_prob_prior, particle, τ, d)
                                            
            return(np.append(trace[-1], np.mean(np.any(acceptance[1:]))))
        
        if paralell:
            res_trial = np.array(Parallel(n_jobs=n_proc)(delayed(move_particle_trial)(i) 
                                           for i in particles_resampled))
        else:    
            res_trial = np.array([move_particle_trial(particle) 
                                  for particle in particles_resampled])
        particles_trial, acc_trial = res_trial[:,0:d], res_trial[:,-1]
        n_steps = int(min(n_step_max,max(2,np.ceil(np.log(1-c) / np.log(1-(np.mean(acc_trial)-1e-6))))))
        
        def move_particle(particle):
            
            trace, acceptance = Gibbs_move(n_steps,
                                            step_sizes, 
                                            log_prob, 
                                            log_prob_prior, particle, τ, d)
            return(np.append(trace[-1], np.mean(np.any(acceptance[1:]))))
        if paralell:
            res = np.array(Parallel(n_jobs=n_proc)(delayed(move_particle)(i) 
                                           for i in particles_trial))
        else:    
            res = np.array([move_particle(particle) for particle in particles_trial])
        particles_moved, acc_rate = res[:,0:d], res[:,-1]
        if verbose:
            print('Generation: ' + str(g) + " ;temperature: "+str(τ_seq[-1])+
                  " ;ESS: "+str(ESS)+
               " ;steps:" + str(n_steps+1) + " ;particle moved: "+
               str(np.mean(acc_rate) * 100) + "%" )
      
        
        cloud = pd.DataFrame(particles_moved)
        cloud.columns = f.parm_names
        # Updating unormalized weights
        cloud['logw'] = logw
        # Updating normalized weights
        cloud['W'] = W
        clouds.append(cloud)

    marginal_log_likelihood = sum([ sp.logsumexp(cloud['logw'] - np.log(popSize)) 
                                            for cloud in clouds[1:g+1]])
    
    log_probs = [log_prob(particle) for particle in particles_moved]
    DIC =  - 2* log_prob(np.mean(particles_moved, axis = 0)) + \
    2* (2* np.mean(log_probs) - 2* log_prob(np.mean(particles_moved, axis = 0)))
    
    f.set_logd(particles_moved)

    logds = np.array([f.logd(x) for x in X])
    WAIC = - 2*( 
        sum(np.log(np.mean(np.exp(logds), axis = 1))) -
                sum(np.var(logds, axis = 1))
                )  
    return(clouds[-1][f.parm_names], marginal_log_likelihood, DIC, WAIC)



