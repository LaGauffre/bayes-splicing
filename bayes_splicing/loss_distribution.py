# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:42:14 2020

@author: pierr
"""
import numpy as np
import numba as nb
import scipy.special as sp
import math as ma
import scipy.stats as st
import scipy as sc

def phi_vec(z):
    """
    Cdf of unit normal distribution

    Parameters
    ----------
    z : Float

    Returns
    -------
    CDF of unit normal distribution
    """
    return( 1 / 2 * (1 + sp.erf(z /np.sqrt(2))))

@nb.jit(nopython=True)
def phi_jit(z):
    """
    Cdf of unit normal distribution

    Parameters
    ----------
    z : Float

    Returns
    -------
    CDF of unit normal distribution
    """
    return( 1 / 2 * (1 + ma.erf(z /np.sqrt(2))))

@nb.jit(nopython = True)
def reg_inc_gamma(a,x):
    """
    Compute the regularized incomplete gamma function.

    Parameters
    ----------
    a : float 
        parameter of the gamma function.
    x : float
        upper bound of the integral.

    Returns
    -------
    float
    Value of the regularized incomplete gamma function
    
    Example
    -------
    a, x = 3, 1 
    sp.gammainc(a, x),  reg_inc_gamma(a,x)
    """
    xam = -x + a * np.log(x)
    if x <= 1 + a:
        s = 1/a
        r = s
        for j in np.arange(1, 61, 1):
            r = r * x / (a + j)
            s = s + r
            if (abs(r /s) < 1e-15):
                break
        gin  = np.exp(xam) * s
        ga = ma.gamma(a)
        gip = gin / ga
        gim = ga - gin
    else:
        t0 = 0
        for j in np.flip(np.arange(1, 61, 1)):
            t0 = (j - a) / (1 + j / (x + t0))
        gim  = np.exp(xam) / (x + t0)
        ga = ma.gamma(a)
        gin = ga - gim
        gip = 1 - gim / ga
    return(gip)

class loss_model:
    def __init__(self, name, parm_names):
        self.name = name
        self.parm_names = parm_names
        self.d= len(parm_names)
        
    def set_logp(self, X):
        if self.name == "Exp":
            def logp(parms):
                λ = parms[0]
                if λ > 0:
                    return(len(X) * np.log(λ) - λ * np.sum(X))
                else:
                    return(-np.inf)
        elif self.name == "Gamma":
            def logp(parms):
                r, m = parms
                if np.all(parms > 0):
                    return(-np.sum(X) / m + (r - 1)* np.sum(np.log(X)) - len(X) * np.log(ma.gamma(r)) - r * len(X) * np.log(m))
                else:
                    return(-np.inf)
                
                
        elif self.name == "Weibull":
            def logp(parms):
                k, β = parms
                if np.all(parms > 0):
                    return(len(X) * (np.log(k) - np.log(β)) + np.sum((k-1) * (np.log(X) -  np.log(β))) - np.sum((X / β)** k))
                else:
                    return(-np.inf)
        elif self.name == "Lognormal":
            def logp(parms):
                μ, σ = parms
                if σ > 0:
                    return(-len(X) * (np.log(σ) + np.log(2 * ma.pi) / 2) - np.sum(np.log(X)) - np.sum((np.log(X) - μ)**2) / 2 / σ **2)
                else:
                    return(-np.inf)
        elif self.name == "Inverse-Gaussian":
            def logp(parms):
                μ, λ = parms
                if np.all(parms > 0):
                    return(len(X) / 2 * (np.log(λ) - np.log(2 * ma.pi) ) - 
                           3 / 2 * np.sum(np.log(X)) - 
                           λ / 2 / μ**2 * np.sum((X - μ)**2 / X))
                else:
                    return(-np.inf)
                
        elif self.name == "Lomax":
            def logp(parms):
                α, σ = parms
                if np.all(parms > 0):
                    return(len(X)*( np.log(α) + α * np.log(σ)) - (α + 1) * np.sum(np.log(X + σ)) )
                else:
                    return(-np.inf)
        
        elif self.name == "Log-Logistic":
            def logp(parms):
                β, σ = parms
                if np.all(parms > 0):
                    return(len(X)*( np.log(β) + β * np.log(σ)) + (β - 1) * np.sum(np.log(X)) -
                           2 * np.sum(np.log(σ**β + X**β)))
                else:
                    return(-np.inf)
                
        elif self.name == "Burr":
            def logp(parms):
                α, β, σ = parms
                if np.all(parms > 0):
                    return(len(X)*(np.log(α) + np.log(β) + α * β * np.log(σ)) + \
                           (β - 1) * np.sum(np.log(X)) - (α + 1) * np.sum(np.log(σ**β + X**β)) )
                else:
                    return(-np.inf)
                
        elif self.name == "Pareto":
            def logp(parms):
                α, γ = parms
                if np.all(parms > 0) and np.all(X >= γ) :
                    
                    return(len(X)*(np.log(α) + α * np.log(γ)) - (α + 1) * np.sum(np.log(X)))
                else:
                    return(-np.inf)
                
        elif self.name == "GPD":
            def logp(parms):
                ξ, σ, γ = parms
                if np.all(parms > 0) and np.all(X >= γ) :
                    
                    return(- len(X)* np.log(σ) -(1 + ξ) / ξ * np.sum(np.log(1+ξ *(X - γ) / σ)))
                else:
                    return(-np.inf)
        elif self.name == "Inverse-Weibull":
            def logp(parms):
                k, β = parms
                if np.all(parms > 0):
                    return(len(X) * (np.log(k) + k * np.log(β)) - (k+1) * np.sum(np.log(X)) - β**k * np.sum((1 / X)** k))
                else:
                    return(-np.inf)
        
                
        elif self.name == "Gamma-Pareto":
            def logp(parms):
                k, α, θ = parms
        
                if np.all(parms > 0):
                    β = θ / (k + α)
                    r = α*ma.gamma(k)*  reg_inc_gamma(k, θ / β) * np.exp(k+α)*(k+α)**(-k) / \
                    (1+ α*ma.gamma(k) * reg_inc_gamma(k, θ / β) * np.exp(k+α)*(k+α)**(-k))
                    if β > 0 and r > 0 and r < 1:
                        X1 = X[X < θ]
                        X2 = X[X >= θ]
                        F1 = reg_inc_gamma(k, θ / β)

                        return(len(X1) * (np.log(r) - np.log(F1) - np.log(ma.gamma(k)) - \
                                          k * np.log(β)) - np.sum(X1) / β +\
                               (k-1) * np.sum(np.log(X1)) + len(X2) *(np.log(1-r) +\
                                np.log(α) + α * np.log(θ)) - (α + 1) * np.sum(np.log(X2))
                               )
                    else: 
                        return(-np.inf)
                else:
                    return(-np.inf)
                
        elif self.name == "Inverse-Gamma":
            def logp(parms):
                r, m = parms
                if np.all(parms > 0):
                    return(-m * np.sum(1 / X)  - (r + 1)* np.sum(np.log(X)) - len(X) * np.log(ma.gamma(r)) + r * len(X) * np.log(m))
                else:
                    return(-np.inf)
                
        elif self.name == "Lognormal-Pareto":
            def logp(parms):
                σ, α, θ = parms
        
                if np.all(parms > 0):
                    μ = np.log(θ) - α * σ**2
                    r = (α * σ  *np.sqrt(2* ma.pi) *phi_jit(α * σ) ) /  \
                        (α * σ  *np.sqrt(2* ma.pi) *phi_jit(α * σ) + np.exp(-(α*σ)**2 / 2))
                    if r > 0 and r < 1:
                        X1 = X[X < θ]
                        X2 = X[X >= θ]
                        F1 = phi_jit(α * σ)

                        return(len(X1) * (np.log(r) - np.log(F1 * σ * np.sqrt(2 * ma.pi)))\
                               - np.sum(np.log(X1)) - np.sum((np.log(X1) - μ)**2) / 2 / σ**2 \
                                   + len(X2) *(np.log(1-r) + np.log(α) + α * np.log(θ))\
                                       - (α + 1) * np.sum(np.log(X2))
                               )
                    else: 
                        return(-np.inf)

                else:
                    return(-np.inf)
                
        elif self.name == "Weibull-Pareto":
            def logp(parms):
                k, α, θ = parms
        
                if np.all(parms > 0):
                    β = (k / (k + α))**(1 / k) * θ
                    r = (α / θ)*(1 - np.exp(-(k + α) / k)) / (α / θ + (k / θ) *\
                                                      np.exp(-(k+α)/k))
                    if β > 0 and r > 0 and r < 1:
                        X1 = X[X < θ]
                        X2 = X[X >= θ]
                        F1 = 1 - np.exp(-(θ / β)**k)
                    
                        return(len(X1) * \
                           ( np.log(r) + np.log(k) - k * np.log(β) ) + \
                           (k-1) * np.sum(np.log(X1)) - np.sum( (X1/ β)**k ) -\
                           len(X1) * np.log(F1) + len(X2) *(np.log(1-r) +\
                        np.log(α) + α * np.log(θ)) - (α + 1) * np.sum(np.log(X2))
                       )
                    else: 
                        return(-np.inf)
            
                else:
                    return(-np.inf)
                
        self.logp = nb.jit(nopython = True)(logp)
        
    
    def set_logps(self):
        if self.name == "Exp":
            def logp_body(X, parms,  γ):
                λ = parms[0]
                F1 = 1-np.exp(- λ * γ)
                if λ > 0 and F1 > 0:
                    return(len(X) * np.log(λ) - λ * np.sum(X) - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                λ = parms[0]
                F2 = 1-np.exp(- λ * γ)
                if λ > 0 and F2 < 1:
                    return(len(X) * np.log(λ) - λ * np.sum(X) - len(X) * np.log(1-F2))
                else:
                    return(-np.inf)
                
        
                
        elif self.name == "Gamma":

            def logp_body(X, parms,  γ):
                r, m = parms
                F1 = reg_inc_gamma(r, γ / m)
                if np.all(parms > 0) and F1 > 0:
                    return(-np.sum(X) / m + (r - 1) * np.sum(np.log(X)) - len(X) * np.log(ma.gamma(r)) - r * len(X) * np.log(m) - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                r, m = parms
                F2 = reg_inc_gamma(r, γ / m)
                if np.all(parms > 0) and F2 < 1:
                    return(-np.sum(X) / m + (r - 1) * np.sum(np.log(X)) - len(X) * np.log(ma.gamma(r)) - r * len(X) * np.log(m) - len(X) * np.log(1 - F2))
                else:
                    return(-np.inf)
        
        elif self.name == "Weibull":

            def logp_body(X, parms,  γ):
                k, β = parms
                F1 = 1 - np.exp(-(γ / β)** k)
                if np.all(parms > 0) and F1 > 0:
                    return(len(X) * (np.log(k) - np.log(β)) + np.sum((k-1) * (np.log(X) -  np.log(β))) - np.sum((X / β)** k) - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                k, β = parms
                F1 = 1 - np.exp(-(γ / β)** k)
                if np.all(parms > 0) and F1 < 1:
                    
                    return(len(X) * (np.log(k) - np.log(β)) + np.sum((k-1) * (np.log(X) -  np.log(β))) - np.sum((X / β)** k) - len(X) * np.log(1 - F1))
                else:
                    return(-np.inf)
                
        elif self.name == "Lognormal":
            def logp_body(X, parms,  γ):
                μ, σ = parms
                F1 = phi_jit( (np.log(γ) - μ) / σ)
                if σ > 0 and F1 > 0:
                    return(-len(X) * (np.log(σ) + np.log(2 * ma.pi) / 2) - np.sum(np.log(X)) - np.sum((np.log(X) - μ)**2) / 2 / σ **2 - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                μ, σ = parms
                F2 = phi_jit( (np.log(γ) - μ)  / σ)
                if σ > 0 and F2 < 1:
                    return(-len(X) * (np.log(σ) + np.log(2 * ma.pi) / 2) - np.sum(np.log(X)) - np.sum((np.log(X) - μ)**2) / 2 / σ **2 - len(X) * np.log(1 - F2))
                else:
                    return(-np.inf)
                
        elif self.name == "Inverse-Gaussian":
            def logp_body(X, parms,  γ):
                μ, λ = parms
                F1 = phi_jit(np.sqrt(λ / γ) * (γ/μ - 1)) + np.exp(2 * λ / μ) * phi_jit(- np.sqrt(λ / γ) * (γ/μ + 1)) 
                if np.all(parms > 0) and F1 > 0:
                    return(len(X) / 2 * (np.log(λ) - np.log(2 * ma.pi) ) - 
                           3 / 2 * np.sum(np.log(X)) - 
                           λ / 2 / μ**2 * np.sum((X - μ)**2 / X) - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                μ, λ = parms
                F2 = phi_jit(np.sqrt(λ / γ) * (γ/μ - 1)) + np.exp(2 * λ / μ) * phi_jit(- np.sqrt(λ / γ) * (γ/μ + 1)) 
                if np.all(parms > 0) and F2 < 1:
                    return(len(X) / 2 * (np.log(λ) - np.log(2 * ma.pi) ) - 
                           3 / 2 * np.sum(np.log(X)) - 
                           λ / 2 / μ**2 * np.sum((X - μ)**2 / X) - len(X) * np.log(1 - F2))
                else:
                    return(-np.inf)
                
        elif self.name == "Lomax":
            def logp_body(X, parms,  γ):
                α, σ = parms
                F1 = 1 - σ**α * (γ + σ)**(-α)
                if np.all(parms > 0) and F1 > 0:
                    return(len(X)*( np.log(α) + α * np.log(σ)) - (α + 1) * np.sum(np.log(X + σ)) - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                α, σ = parms
                F1 = 1 - σ**α * (γ + σ)**(-α)
                if np.all(parms > 0) and F1 < 1:
                    
                    return(len(X)*( np.log(α) + α * np.log(σ)) - (α + 1) * np.sum(np.log(X + σ))  - len(X) * np.log(1 - F1))
                else:
                    return(-np.inf)
        
        elif self.name == "Log-Logistic":
            def logp_body(X, parms,  γ):
                β, σ = parms
                F1 = 1 / (1 + (γ / σ)**(-β))
                if np.all(parms > 0) and F1 > 0:
                    return(len(X)*( np.log(β) + β * np.log(σ)) + (β - 1) * np.sum(np.log(X)) -
                           2 * np.sum(np.log(σ**β + X**β)) - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                β, σ = parms
                F1 = 1 / (1 + (γ / σ)**(-β))
                if np.all(parms > 0) and F1 < 1:
                    
                    return(len(X)*( np.log(β) + β * np.log(σ)) + (β - 1) * np.sum(np.log(X)) -
                           2 * np.sum(np.log(σ**β + X**β)) - len(X) * np.log(1 - F1))
                else:
                    return(-np.inf)
        
        elif self.name == "Burr":
            def logp_body(X, parms,  γ):
                α, β, σ = parms
                F1 = 1 - (1 + (γ/σ)**β)**(-α)
                if np.all(parms > 0) and F1 > 0:
                    return(len(X)*(np.log(α) + np.log(β) + α * β * np.log(σ)) + \
                           (β - 1) * np.sum(np.log(X)) - (α + 1) * np.sum(np.log(σ**β + X**β)) \
                           - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                α, β, σ = parms
                F2 = 1 - (1 + (γ/σ)**β)**(-α)
                if np.all(parms > 0) and F2 < 1:
                    
                    return(len(X)*(np.log(α) + np.log(β) + α * β * np.log(σ)) + \
                           (β - 1) * np.sum(np.log(X)) - (α + 1) * np.sum(np.log(σ**β + X**β)) \
                           - len(X) * np.log(1 - F2))
                else:
                    return(-np.inf)
        
        elif self.name == "Pareto-Tail":
            def logp_body(X, parms,  γ):
                return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                α = parms[0]
                if α > 0:
                    
                    return(len(X)*( np.log(α) + α * np.log(γ)) - (α + 1) * np.sum(np.log(X)) )
                else:
                    return(-np.inf)
        elif self.name == "GPD-Tail":
            def logp_body(X, parms,  γ):
                return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                ξ, σ  = parms
                if np.all(parms > 0):
                    return(- len(X)* np.log(σ) -(1 + ξ) / ξ * np.sum(np.log(1+ξ *(X - γ) / σ)))
                else:
                    return(-np.inf)
        elif self.name == "Inverse-Gamma":

            def logp_body(X, parms,  γ):
                r, m = parms
                F1 = 1-reg_inc_gamma(r, m / γ )
                if np.all(parms > 0) and F1 > 0:
                    return(-m * np.sum(1 / X)  - (r + 1)* np.sum(np.log(X)) - len(X) * np.log(ma.gamma(r)) + r * len(X) * np.log(m) - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                r, m = parms
                F2 = 1-reg_inc_gamma(r, m / γ )
                if np.all(parms > 0) and F2 < 1:
                    return(-m * np.sum(1 / X)  - (r + 1)* np.sum(np.log(X)) - len(X) * np.log(ma.gamma(r)) + r * len(X) * np.log(m) - len(X) * np.log(1 - F2))
                else:
                    return(-np.inf)
        elif self.name == "Inverse-Weibull":
            def logp_body(X, parms,  γ):
                k, β = parms
                F1 = np.exp(-(β / γ)** k)
                if np.all(parms > 0) and F1 > 0:
                    return(len(X) * (np.log(k) + k * np.log(β)) - (k+1) * np.sum(np.log(X)) - β**k * np.sum((1 / X)** k) - len(X) * np.log(F1))
                else:
                    return(-np.inf)
            
            def logp_tail(X, parms,  γ):
                k, β = parms
                F2 = np.exp(-(β / γ)** k)
                if np.all(parms > 0) and F2 < 1:
                    return(len(X) * (np.log(k) + k * np.log(β)) - (k+1) * np.sum(np.log(X)) - β**k * np.sum((1 / X)** k) - len(X) * np.log(1 - F2))
                else:
                    return(-np.inf)
        
        self.logp_body = nb.jit(nopython = True)(logp_body)
        self.logp_tail = nb.jit(nopython = True)(logp_tail)
    
    def set_logd(self, parms):
        if self.name == "Exp":
            def logd(x):
                λ = parms[:,0]
                res = np.zeros(len(λ))
                s = λ > 0
                res[np.where(s)] = np.log(λ[s]) - λ[s] * x
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
        elif self.name == "Gamma":
            r , m = parms[:,0], parms[:,1]
            def logd(x):
                
                res = np.zeros(len(r))
                s = np.logical_and(r > 0, m > 0)
                res[np.where(s)] = -x / m[s] + (r[s] - 1)* np.sum(np.log(x)) -  np.log(sp.gamma(r[s])) - r[s] * np.log(m[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
        elif self.name == "Weibull":
            def logd(x):
                k, β = parms[:,0], parms[:,1]
                res = np.zeros(len(k))
                s = np.logical_and(k > 0, β > 0)
                res[np.where(s)] = np.log(k[s]) -np.log(β[s]) + (k[s]-1) * np.log(x / β[s]) - (x / β[s])** k[s]
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        elif self.name == "Lognormal":
            def logd(x):
                μ, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(μ))
                s = σ > 0
                res[np.where(s)] =  - np.log(σ[s]) - np.log(2 * ma.pi) / 2 - np.log(x) - ((np.log(x) - μ[s])**2) / 2 / σ[s] **2
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
        elif self.name == "Inverse-Gaussian":
            def logd(x):
                μ, λ = parms[:,0], parms[:,1]
                res = np.zeros(len(μ))
                s = np.logical_and(μ > 0, λ > 0)
                res[np.where(s)] =  (np.log(λ[s]) - np.log(2 * ma.pi)) / 2  - 3 / 2 * np.log(x) - λ[s] / 2 / μ[s]**2 * (x - μ[s])**2 / x
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        elif self.name == "Lomax":
            def logd(x):
                α, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(α))
                s = np.logical_and(α > 0, σ > 0)
                res[np.where(s)] = np.log(α[s]) + α[s] * np.log(σ[s]) - (α[s] + 1) * np.log(x + σ[s]) 
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "Log-Logistic":
            def logd(x):
                β, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(β))
                s = np.logical_and(β > 0, σ > 0)
                res[np.where(s)] = (np.log(β[s]) + β[s] * np.log(σ[s])) + (β[s] - 1) * np.log(x) - \
                           2 * np.log(σ[s]**β[s] + x**β[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "Burr":
            def logd(x):
                α, β, σ = parms[:,0], parms[:,1], parms[:,2]
                res = np.zeros(len(β))
                s = np.logical_and(α >0, np.logical_and(β > 0, σ > 0))
                res[np.where(s)] = np.log(α[s]) + np.log(β[s]) + α[s] * β[s] * np.log(σ[s]) +\
                        (β[s] - 1) * np.log(x) - (α[s] + 1) * np.log(σ[s]**β[s] + x**β[s]) 
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "Pareto":
            def logd(x):
                α, γ = parms[:,0], parms[:,1]
                res = np.zeros(len(α))
                s = np.logical_and(np.logical_and(α > 0, γ > 0), γ <= x)
                res[np.where(s)] = np.log(α[s]) + α[s] * np.log(γ[s]) - (α[s] + 1) * np.log(x) 
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "GPD":
            def logd(x):
                ξ, σ, γ = parms[:,0], parms[:,1], parms[:,2]
                res = np.zeros(len(ξ))
                s = np.logical_and(np.logical_and(np.logical_and(ξ > 0, σ > 0), γ <= x), γ > 0)
                res[np.where(s)] = - np.log(σ[s]) -(1 + ξ[s]) / ξ[s] * np.log(1+ξ[s] *(x - γ[s]) / σ[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
        elif self.name == "Inverse-Gamma":
            r , m = parms[:,0], parms[:,1]
            def logd(x):
                
                res = np.zeros(len(r))
                s = np.logical_and(r > 0, m > 0)
                res[np.where(s)] = -m[s] / x  - (r[s] + 1)* np.log(x) - np.log(sp.gamma(r[s])) + r[s] * np.log(m[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        elif self.name == "Inverse-Weibull":
            def logd(x):
                k, β = parms[:,0], parms[:,1]
                res = np.zeros(len(k))
                s = np.logical_and(k > 0, β > 0)
                res[np.where(s)] = np.log(k[s]) + k[s] * np.log(β[s]) - (k[s] + 1) * np.log(x) - (β[s] / x)** k[s]
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "Gamma-Pareto":
            k, α, θ = parms[:,0], parms[:,1], parms[:,2]
            β, r, F1 = np.zeros(len(α)), np.zeros(len(α)), np.zeros(len(α))
            s0 = np.logical_and(np.logical_and(k > 0, θ > 0),α > 0)
            β[s0] = θ[s0] / (k[s0] + α[s0])
            r[s0] = α[s0]*sp.gamma(k[s0])*  sp.gammainc(k[s0],θ[s0] / β[s0]) * np.exp(k[s0]+α[s0])*(k[s0]+α[s0])**(-k[s0]) / \
            (1+ α[s0]*sp.gamma(k[s0]) * sp.gammainc(k[s0], θ[s0] / β[s0]) * np.exp(k[s0]+α[s0])*(k[s0]+α[s0])**(-k[s0]))
            F1[s0] = sp.gammainc(k[s0], θ[s0] / β[s0])
            def logd(x):
                res = np.zeros(len(α))
                s = np.logical_and(np.logical_and(β > 0, r > 0), r < 1)
                s1 = np.logical_and(s, x < θ)
                s2 = np.logical_and(s, x >= θ)

                res1 = np.log(r[s1]) - np.log(F1[s1]) - np.log(sp.gamma(k[s1])) - \
                    k[s1] * np.log(β[s1]) - x / β[s1] + (k[s1]-1) * np.log(x)

                res2 = (np.log(1-r[s2]) + np.log(α[s2]) + α[s2] * \
                        np.log(θ[s2])) - (α[s2] + 1) * np.log(x)

                res[np.where(s1)] = res1
                res[np.where(s2)] = res2
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "Lognormal-Pareto":
            σ, α, θ = parms[:,0], parms[:,1], parms[:,2]
            μ = np.log(θ) - α * σ**2
            r = (α * σ  * np.sqrt(2* ma.pi) *phi_vec(α * σ) ) /  \
                (α * σ  * np.sqrt(2* ma.pi) *phi_vec(α * σ) + np.exp(-(α*σ)**2 / 2))
            F1 = phi_vec(α * σ)
            def logd(x):

                s = np.logical_and(r > 0, r < 1)
                s1 = np.logical_and(s, x < θ)
                s2 = np.logical_and(s, x >= θ)
                res = np.zeros(len(r))

                res1 = (np.log(r[s1]) - np.log(F1[s1] * σ[s1] * np.sqrt(2 * ma.pi)))\
                                - np.log(x) - (np.log(x) - μ[s1])**2 / 2 / σ[s1]**2

                res2 = (np.log(1-r[s2]) + np.log(α[s2]) + α[s2] * np.log(θ[s2])) - (α[s2] + 1) * np.log(x)
                res[np.where(s1)] = res1
                res[np.where(s2)] = res2
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "Weibull-Pareto":
            k, α, θ = parms[:,0], parms[:,1], parms[:,2]
            F1, β, r = np.zeros(len(α)), np.zeros(len(α)), np.zeros(len(α))
            s0 = np.logical_and(np.logical_and(k > 0, α > 0), θ > 0)    
            β[s0] = (k[s0] / (k[s0] + α[s0]))**(1 / k[s0]) * θ[s0]
            r[s0] = (α[s0] / θ[s0])*(1 - np.exp(-(k[s0] + α[s0]) / k[s0])) / (α[s0] / θ[s0] + (k[s0] / θ[s0]) * \
                                                              np.exp(-(k[s0] + α[s0]) / k[s0]))
            s00 = np.logical_and(s0, β > 1e-300)
            F1[s00] = 1 - np.exp(-(θ[s00] / β[s00])**k[s00])
            def logd(x):
                res = np.zeros(len(α))
                s = np.logical_and(np.logical_and(β > 1e-300, r > 0), r < 1)
                s1 = np.logical_and(s, x < θ)
                s2 = np.logical_and(s, x >= θ)


                res1 = (np.log(r[s1]) + np.log(k[s1]) - k[s1] * np.log(β[s1])) + \
                    (k[s1]-1) * np.log(x) -  (x/ β[s1]) ** k[s1] - \
                        np.log(F1[s1])

                res2 = (np.log(1-r[s2]) + np.log(α[s2]) + α[s2] * \
                        np.log(θ[s2])) - (α[s2] + 1) * np.log(x)

                res[np.where(s1)] = res1
                res[np.where(s2)] = res2
                res[np.where(np.invert(s))] = - np.inf
                return(res)
            
        self.logd = logd
        
            
    
    def set_logds(self):
        if self.name == "Exp":
            def logd_body(x, parms, γ):
                λ = parms[:,0]
                res = np.zeros(len(λ))
                s = np.logical_and(λ > 0, x < γ)
                res[np.where(s)] = np.log(λ[s])- λ[s] * x - np.log(1-np.exp(- λ[s] * γ[s]))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms, γ):
                λ = parms[:,0]
                res = np.zeros(len(λ))
                s = np.logical_and(λ > 0, x >= γ)
                res[np.where(s)] = np.log(λ[s]) - λ[s] * (x - γ[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
        elif self.name == "Gamma":
            
            def logd_body(x, parms,  γ):
                r , m = parms[:,0], parms[:,1]
                res = np.zeros(len(r))
                s = np.logical_and(np.logical_and(r > 0, m > 0), x < γ)
                res[np.where(s)] = -x / m[s] + (r[s] - 1)* np.sum(np.log(x)) -  np.log(sp.gamma(r[s])) - r[s] * np.log(m[s]) - np.log(sp.gammainc(r[s], γ[s] / m[s]))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms,  γ):
                r , m = parms[:,0], parms[:,1]
                res = np.zeros(len(r))
                s = np.logical_and(np.logical_and(r > 0, m > 0), x >= γ)
                res[np.where(s)] = -x / m[s] + (r[s] - 1)* np.sum(np.log(x)) -  np.log(sp.gamma(r[s])) - r[s] * np.log(m[s]) - np.log(1 - sp.gammainc(r[s], γ[s] / m[s]))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            
            
        elif self.name == "Weibull":
            def logd_body(x, parms,  γ):
                k, β = parms[:,0], parms[:,1]
                res = np.zeros(len(k))
                s = np.logical_and(np.logical_and(k > 0, β > 0), x < γ)
                res[np.where(s)] = np.log(k[s]) -np.log(β[s]) + (k[s]-1) * np.log(x / β[s]) - (x / β[s])** k[s] - np.log(1 - np.exp(-(γ[s] / β[s])** k[s]))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms,  γ):
                k, β = parms[:,0], parms[:,1]
                res = np.zeros(len(k))
                s = np.logical_and(np.logical_and(k > 0, β > 0), x >= γ)
                res[np.where(s)] = np.log(k[s]) -np.log(β[s]) + (k[s]-1) * np.log(x / β[s]) - (x / β[s])** k[s] + (γ[s] / β[s])** k[s]
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        elif self.name == "Lognormal":
            
            def logd_body(x, parms,  γ):
                μ, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(μ))
                F1 = phi_vec((np.log(γ) - μ)  / σ)
                s = np.logical_and(np.logical_and(σ > 0 , x < γ), F1 > 0)
                res[np.where(s)] = - np.log(σ[s]) - np.log(2 * ma.pi) / 2 - np.log(x) - ((np.log(x) - μ[s])**2) / 2 / σ[s] **2 - np.log(F1[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms,  γ):
                μ, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(μ))
                F2 = phi_vec((np.log(γ) - μ)  / σ)
                s = np.logical_and(np.logical_and(σ > 0 , x >= γ), F2 < 1 ) 
                res[np.where(s)] = - np.log(σ[s]) - np.log(2 * ma.pi) / 2 - np.log(x) - ((np.log(x) - μ[s])**2) / 2 / σ[s] **2 - np.log(1- F2[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
        elif self.name == "Inverse-Gaussian":
            
            def logd_body(x, parms,  γ):
                μ, λ = parms[:,0], parms[:,1]
                F1 = phi_vec(np.sqrt(λ / γ) * (γ/μ - 1)) + np.exp(2 * λ / μ) * phi_vec(- np.sqrt(λ / γ) * (γ/μ + 1)) 
                res = np.zeros(len(μ))
                s = np.logical_and(np.logical_and(np.logical_and(μ > 0, λ > 0) , x < γ), F1 > 0 )
                res[np.where(s)] = (np.log(λ[s]) - np.log(2 * ma.pi)) / 2  - 3 / 2 * np.log(x) - λ[s] / 2 / μ[s]**2 * (x - μ[s])**2 / x - np.log(F1[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms,  γ):
                μ, λ = parms[:,0], parms[:,1]
                F2 = phi_vec(np.sqrt(λ / γ) * (γ/μ - 1)) + np.exp(2 * λ / μ) * phi_vec(- np.sqrt(λ / γ) * (γ/μ + 1)) 
                res = np.zeros(len(μ))
                s = np.logical_and(np.logical_and(np.logical_and(μ > 0, λ > 0) , x >= γ), F2 < 1 )
                res[np.where(s)] = (np.log(λ[s]) - np.log(2 * ma.pi)) / 2  - 3 / 2 * np.log(x) - λ[s] / 2 / μ[s]**2 * (x - μ[s])**2 / x - np.log(1 - F2[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        elif self.name == "Lomax":
            
            def logd_body(x, parms,  γ):
                α, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(α))
                s = np.logical_and(np.logical_and(α > 0, σ > 0), x < γ)
                res[np.where(s)] = np.log(α[s]) + α[s] * np.log(σ[s]) - (α[s] + 1) * np.log(x + σ[s]) - np.log(1 - σ[s]**α[s] * (γ[s] + σ[s])**(-α[s]))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms,  γ):
                α, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(α))
                s = np.logical_and(np.logical_and(α > 0, σ > 0), x >= γ)
                res[np.where(s)] = np.log(α[s]) + α[s] * np.log(σ[s]) - (α[s] + 1) * np.log(x + σ[s]) - np.log(σ[s]**α[s] * (γ[s] + σ[s])**(-α[s]))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "Log-Logistic":
            
            def logd_body(x, parms,  γ):
                β, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(β))
                s = np.logical_and(np.logical_and(β > 0, σ > 0), x < γ)
                res[np.where(s)] = (np.log(β[s]) + β[s] * np.log(σ[s])) + (β[s] - 1) * np.log(x) - \
                           2 * np.log(σ[s]**β[s] + x**β[s]) - np.log(1 / (1 + (γ[s] / σ[s])**(-β[s])))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms,  γ):
                β, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(β))
                s = np.logical_and(np.logical_and(β > 0, σ > 0), x >= γ)
                res[np.where(s)] = (np.log(β[s]) + β[s] * np.log(σ[s])) + (β[s] - 1) * np.log(x) - \
                           2 * np.log(σ[s]**β[s] + x**β[s]) - np.log(1 - 1 / (1 + (γ[s] / σ[s])**(-β[s])))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "Burr":
            
            def logd_body(x, parms,  γ):
                α, β, σ = parms[:,0], parms[:,1], parms[:,2]
                F1 = 1 - (1 + (γ/σ)**β)**(-α)
                res = np.zeros(len(β))
                s = np.logical_and(np.logical_and(α >0, np.logical_and(β > 0, σ > 0)), x < γ)
                res[np.where(s)] = np.log(α[s]) + np.log(β[s]) + α[s] * β[s] * np.log(σ[s]) +\
                        (β[s] - 1) * np.log(x) - (α[s] + 1) * np.log(σ[s]**β[s] + x**β[s]) - np.log(F1[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms,  γ):
                α, β, σ = parms[:,0], parms[:,1], parms[:,2]
                F2 = 1 - (1 + (γ/σ)**β)**(-α)
                res = np.zeros(len(β))
                s = np.logical_and(np.logical_and(α >0, np.logical_and(β > 0, σ > 0)), x >= γ)
                res[np.where(s)] = np.log(α[s]) + np.log(β[s]) + α[s] * β[s] * np.log(σ[s]) +\
                        (β[s] - 1) * np.log(x) - (α[s] + 1) * np.log(σ[s]**β[s] + x**β[s]) - np.log(1 - F2[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        elif self.name == "Pareto-Tail":
            
            def logd_body(x, parms,  γ):
                return(None)
            
            def logd_tail(x, parms,  γ):
                α = parms[:,0]
                res = np.zeros(len(α))
                s = np.logical_and(α > 0, x >= γ)
                res[np.where(s)] = np.log(α[s]) + α[s] * np.log(γ[s]) - (α[s] + 1) * np.log(x)
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        elif self.name == "GPD-Tail":
            
            def logd_body(x, parms,  γ):
                return(None)
            
            def logd_tail(x, parms,  γ):
                ξ, σ = parms[:,0], parms[:,1]
                res = np.zeros(len(ξ))
                s = np.logical_and(np.logical_and(ξ > 0, σ), x >= γ)
                res[np.where(s)] = - np.log(σ[s]) -(1 + ξ[s]) / ξ[s] * np.log(1+ξ[s] *(x - γ[s]) / σ[s])
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        elif self.name == "Inverse-Gamma":
            
            def logd_body(x, parms,  γ):
                r , m = parms[:,0], parms[:,1]
                res = np.zeros(len(r))
                s = np.logical_and(np.logical_and(r > 0, m > 0), x < γ)
                res[np.where(s)] = -m[s] / x  - (r[s] + 1)* np.log(x) - np.log(sp.gamma(r[s])) + r[s] * np.log(m[s]) - np.log(1-sp.gammainc(r[s], m[s] / γ[s]))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms,  γ):
                r , m = parms[:,0], parms[:,1]
                res = np.zeros(len(r))
                s = np.logical_and(np.logical_and(r > 0, m > 0), x >= γ)
                res[np.where(s)] = -m[s] / x  - (r[s] + 1)* np.log(x) - np.log(sp.gamma(r[s])) + r[s] * np.log(m[s]) - np.log(sp.gammainc(r[s], m[s] / γ[s]))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
        elif self.name == "Inverse-Weibull":
            
            def logd_body(x, parms,  γ):
                k, β = parms[:,0], parms[:,1]
                res = np.zeros(len(k))
                s = np.logical_and(np.logical_and(k > 0, β > 0), x < γ)
                res[np.where(s)] = np.log(k[s]) + k[s] * np.log(β[s]) - (k[s] + 1) * np.log(x) - (β[s] / x)** k[s] + (β[s] / γ[s])** k[s]
                res[np.where(np.invert(s))] = -np.inf
                return(res)
            
            def logd_tail(x, parms,  γ):
                k, β = parms[:,0], parms[:,1]
                res = np.zeros(len(k))
                s = np.logical_and(np.logical_and(k > 0, β > 0), x >= γ)
                res[np.where(s)] = np.log(k[s]) + k[s] * np.log(β[s]) - (k[s] + 1) * np.log(x) - (β[s] / x)** k[s] -np.log(1-np.exp(-(β[s] / γ[s])** k[s]))
                res[np.where(np.invert(s))] = -np.inf
                return(res)
        
        
        self.logd_body = logd_body
        self.logd_tail = logd_tail
        
                
    def sample(self, parms, n):
        if self.name == "Exp":
            λ = parms[0]
            return(np.random.exponential(λ, size = n))
        
        elif self.name == "Burr":
            α, β, σ = parms
            return(st.burr12( β, α).rvs(size = n) * σ)
        
        elif self.name == "Gamma":
            r , m = parms
            return(st.gamma(r).rvs(size = n) * m)
        elif self.name == "Inverse-Gaussian":
            μ, λ = parms
            res = np.zeros(n)
            y, u = (np.random.normal(size = n))**2, np.random.uniform(size = n)
            x = μ + μ**2 * y / 2 / λ - μ / 2 / λ * np.sqrt(4 * μ * λ * y + μ**2 * y**2)
            s = u < μ / (μ + x)
            res[s] = x[s]
            res[np.invert(s)] = μ**2 / x[np.invert(s)]
            return(res)
        
        elif self.name == "Log-Logistic":
            β, σ = parms
            return(st.fisk(β).rvs(size = n) * σ)
        elif self.name == "Lognormal":
            μ, σ = parms
            return(np.exp(st.norm().rvs(n) * σ + μ))
        
        elif self.name == "Gamma-Pareto":
            k, α, θ = parms
            β = θ / (k + α)
            r = α*sp.gamma(k)*  sp.gammainc(k,  θ / β) * np.exp(k+α)*(k+α)**(-k) / \
            (1+ α*sp.gamma(k) * sp.gammainc(k, θ / β) * np.exp(k+α)*(k+α)**(-k))
    
            gamma_rv = st.gamma(k)
            par_rv = st.pareto(α)
            binom_rv =  st.binom(1, r)
            par_rvs = θ * par_rv.rvs(size = n)
            binom_rvs = binom_rv.rvs(size = n)
            gamma_rvs = β * gamma_rv.ppf(sp.gammainc(k, θ / β) *\
                               np.random.uniform(size = n))
            return(binom_rvs * gamma_rvs + (1 - binom_rvs) * par_rvs)
        
        elif self.name == "Lognormal-Pareto":
            σ, α, θ = parms
            μ = np.log(θ) - α * σ**2
    
            r = (α * σ  *np.sqrt(2* ma.pi) *phi_jit(α * σ) ) /  \
                (α * σ  *np.sqrt(2* ma.pi) *phi_jit(α * σ) + np.exp(-(α*σ)**2 / 2)) 
    
            lnorm_rv = st.lognorm(s = σ, scale = np.exp(μ))
    
            par_rv = st.pareto(α)
            binom_rv =  st.binom(1, r)
            par_rvs = θ * par_rv.rvs(size = n)
            binom_rvs = binom_rv.rvs(size = n)
            lnorm_rvs = lnorm_rv.ppf(lnorm_rv.cdf(θ) *\
                                       np.random.uniform(size = n))
            return(binom_rvs * lnorm_rvs + (1 - binom_rvs) * par_rvs)
        
        elif self.name == "Weibull-Pareto":
            k, α, θ = parms
            β = (k / (k + α))**(1 / k) * θ
            r = (α / θ)*(1 - np.exp(-(k + α) / k))\
            / (α / θ + (k / θ)*np.exp(-(k + α) / k))
            weib_rv = st.weibull_min(k)
            par_rv = st.pareto(α)
            binom_rv =  st.binom(1, r)
            par_rvs = θ * par_rv.rvs(size = n)
            binom_rvs = binom_rv.rvs(size = n)
            weib_rvs = β * weib_rv.ppf(weib_rv.cdf(θ / β) *\
                                       np.random.uniform(size = n))
            return(binom_rvs * weib_rvs + (1 - binom_rvs) * par_rvs)
        elif self.name == "Pareto":
            α, γ = parms
            return( γ * st.pareto(α).rvs(n))
        elif self.name == "GPD":
            ξ, σ, γ  = parms
            return( γ + σ * st.genpareto(ξ).rvs(n))
        
        elif self.name == "Inverse-Gamma":
            r , m = parms
            return(st.invgamma(r).rvs(size = n) * m)
        elif self.name == "Inverse-Weibull":
            k, β = parms
            return(  st.invweibull(k).rvs(size = n) * β )
        
    
    def set_cdf(self):
        if self.name == "Exp":
            def cdf(parms, x):
                λ = parms[0]
                return(1 - np.exp(- λ * x))
        elif self.name == "Gamma":
            def cdf(parms, x):
                r , m = parms
                return(reg_inc_gamma(r, x / m))
        elif self.name == "Inverse-Gaussian":
            def cdf(parms, x):
                μ, λ = parms
                return( phi_jit(np.sqrt(λ / x) * (x / μ - 1)) + 
                   np.exp(2 * λ / μ) * 
                   phi_jit(- np.sqrt(λ / x) * (x / μ + 1) ))
        elif self.name == "Lognormal":
            def cdf(parms, x):
                μ, σ  = parms
                return(phi_jit( (np.log(x) - μ) / σ))
        elif self.name == "Weibull":
            def cdf(parms, x):
                k, β  = parms
                return(1 - np.exp(-(x / β)** k))
        elif self.name == "Lomax":
            def cdf(parms, x):
                α, σ = parms
                return(1 - (σ / (x + σ))**α)
        elif self.name == "Log-Logistic":
            def cdf(parms, x):
                β, σ = parms
                return(1 / (1 + (x / σ)**(-β)))
        elif self.name == "Burr":
            def cdf(parms, x):
                α, β, σ = parms
                return(1 - (1 + (x / σ)**β)**(-α))
        elif self.name == "Pareto-Tail":
            def cdf(parms, γ, x):
                α = parms[0]
                return(1 - (γ / x)**α)
        elif self.name == "Pareto":
            def cdf(parms, x):
                α, γ = parms
                return(1 - (γ / x)**α)
        elif self.name == "GPD":
            def cdf(parms, x):
                ξ, σ, γ  = parms
                return(1 - (1+ ξ*(x - γ) / σ )**(-1 / ξ) )
        elif self.name == "GPD-Tail":
            def cdf(parms, γ, x):
                ξ, σ  = parms
                return(1 - (1+ ξ*(x - γ) / σ )**(-1 / ξ) )
        elif self.name == "Inverse-Gamma":
            def cdf(parms, x):
                r , m = parms
                return(1-reg_inc_gamma(r, m / x))
        elif self.name == "Inverse-Weibull":
            def cdf(parms, x):
                k, β  = parms
                return(np.exp(-(β / x)** k))
        
        
        elif self.name == "Gamma-Pareto":
            def cdf(parms, x):
                k, α, θ = parms
                β = θ / (k + α)
                cdf2 = reg_inc_gamma(k,  θ / β)
                r = α*ma.gamma(k)*  cdf2 * np.exp(k+α)*(k+α)**(-k) / \
                (1+ α*ma.gamma(k) * cdf2 * np.exp(k+α)*(k+α)**(-k))
                cdf1 = reg_inc_gamma(k,  x / β)
                if x < θ: 
                    return(r * cdf1 / cdf2 ) 
                else:
                    return(r + (1-r) * (1 - (θ / x)**(α)))
        elif self.name == "Lognormal-Pareto":
            def cdf(parms, x):
                σ, α, θ = parms
                μ = np.log(θ) - α * σ**2
                r = (α * σ  *np.sqrt(2* ma.pi) * phi_jit(α * σ) ) /  \
                (α * σ  *np.sqrt(2* ma.pi) * phi_jit(α * σ) + np.exp(-(α*σ)**2 / 2)) 
                cdf2 = 1/2 + ma.erf( (np.log(θ) - μ) / σ / np.sqrt(2)  ) / 2
                cdf1 = 1/2 + ma.erf( (np.log(x) - μ) / σ / np.sqrt(2)  ) / 2
                if x < θ: 
                    return(r * cdf1 / cdf2 ) 
                else:
                    return(r + (1-r) * (1 - (θ / x)**(α)))
        
        elif self.name == "Weibull-Pareto":
            def cdf(parms, x):
                k, α, θ = parms
                β = (k / (k + α))**(1 / k) * θ
                r = (α / θ) * (1 - np.exp(-(k + α) / k))\
                / (α / θ + (k / θ) * np.exp(-(k + α) / k))
    
                cdf2 = 1-np.exp(-(θ / β ) ** k )
                cdf1 = 1-np.exp(-(x / β ) ** k )
                if x < θ: 
                    return(r * cdf1 / cdf2 ) 
                else:
                    return(r + (1-r) * (1 - (θ / x)**(α)))
        
        self.cdf = nb.jit(nopython = True)(cdf)
        
        
    
    def set_pdf(self):
        if self.name == "Exp":
            def pdf(parms, x):
                λ = parms[0]
                return(λ * np.exp(- λ * x))
        elif self.name == "Gamma":
            def pdf(parms, x):
                r , m = parms
                return(x**(r - 1) * np.exp(- x / m) / ma.gamma(r) / m**r )
        elif self.name == "Inverse-Gaussian":
            def pdf(parms, x):
                μ, λ = parms
                return(np.sqrt(λ / 2 / ma.pi / x**3) * np.exp(- λ *(x - μ)**2 /2 / μ**2 / x ))
        
        elif self.name == "Lognormal":
            def pdf(parms, x):
                μ, σ  = parms
                return(np.exp(- (np.log(x) - μ)**2 / 2 / σ**2 ) / x / σ / np.sqrt(2 * ma.pi))
        elif self.name == "Weibull":
            def pdf(parms, x):
                k, β  = parms
                return((k / β) * (x / β)**(k-1) * np.exp(-(x / β)**k))
        elif self.name == "Lomax":
            def pdf(parms, x):
                α, σ = parms
                return(α * σ**α / (x + σ)**(α + 1) )
        elif self.name == "Log-Logistic":
            def pdf(parms, x):
                β, σ = parms
                return(β * σ**β * x**(β - 1) / (σ**β + x**β)**2)
        elif self.name == "Burr":
            def pdf(parms, x):
                α, β, σ = parms
                return(α * β * σ**(α * β) * x**(β - 1) / (σ**β + x**β)**(α + 1))
        
        elif self.name == "Pareto-Tail":
            def pdf(parms, γ,  x):
                α = parms[0]
                return(α * γ**α / x**(α + 1))
        elif self.name == "Pareto":
            def pdf(parms, x):
                α, γ = parms
                return(α * γ**α / x**(α + 1))
        elif self.name == "GPD":
            def pdf(parms, x):
                ξ, σ, γ  = parms
                return(1 / σ * (1+ ξ*(x - γ) / σ )**(-(1+ξ) / ξ) )
        elif self.name == "GPD-Tail":
            def pdf(parms, γ, x):
                ξ, σ  = parms
                return(1 / σ * (1+ ξ*(x - γ) / σ )**(-(1+ξ) / ξ) )
        elif self.name == "Inverse-Gamma":
            def pdf(parms, x):
                r , m = parms
                return((1 / x)**(r + 1) * np.exp(- m / x) / ma.gamma(r) * m**r )
        elif self.name == "Inverse-Weibull":
            def pdf(parms, x):
                k, β  = parms
                return((k / x) * (β / x)**k * np.exp(-(β / x)**k))

        self.pdf = nb.jit(nopython = True)(pdf)
    
    def set_ppf(self):     
        if self.name == "Exp":
            def ppf(parms, y):
                λ = parms[0]
                return( -np.log(1-y) / λ )
        elif self.name == "Gamma":
            def ppf(parms, y):
                r , m = parms
                return(st.gamma(r).ppf(y) * m)
        elif self.name == "Weibull":
            def ppf(parms, y):
                k, β = parms
                return( β * (-np.log(1-y))**(1/k) )
        elif self.name == "Lognormal":
            def ppf(parms, y):
                μ, σ  = parms
                return( np.exp(st.norm().ppf(y) * σ + μ))
        
        elif self.name == "Lomax":
            def ppf(parms, y):
                α, σ = parms
                return( σ * ((1-y)**(-1 / α) - 1))
            
        elif self.name == "Log-Logistic":
            def ppf(parms, y):
                β, σ = parms
                return( σ * (1 / y - 1)**(-1/β))
        elif self.name == "Burr":
            def ppf(parms, y):
                α, β, σ = parms
                return( σ * ( (1-y)**(-1 / α) - 1)**(1 / β))
        elif self.name == "Pareto-Tail":
            def ppf(parms, γ, y):
                α = parms[0]
                return( γ * (1-y)**(-1 / α))
        elif self.name == "Pareto":
            def ppf(parms, y):
                α, γ = parms
                return( γ * ((1-y)**(-1 / α)))
        elif self.name == "GPD":
            def ppf(parms, y):
                ξ, σ, γ  = parms
                return( γ + σ / ξ * ((1 - y)**(-ξ) - 1) )
        elif self.name == "GPD-Tail":
            def ppf(parms, γ, y):
                ξ, σ = parms
                return(γ + σ / ξ * ((1 - y)**(-ξ) - 1))
        elif self.name == "Inverse-Gamma":
            def ppf(parms, y):
                r , m = parms
                return(st.invgamma(r).ppf(y) * m)
        elif self.name == "Inverse-Weibull":
            def ppf(parms, y):
                k, β = parms
                return( β * (-np.log(y))**(-1/k) )
        elif self.name == "Inverse-Gaussian":
            def ppf(parms, y):
                μ,  λ = parms
                return(st.invgauss(μ / λ).ppf(y) * λ)
        
        elif self.name == "Weibull-Pareto" or self.name == "Gamma-Pareto" \
            or self.name == "Lognormal-Pareto":
            self.set_cdf()
            def ppf(parms, y):
                
                low, up = 0, 1e8
                err = 1e-6
                while up - low > err:
                    new = (up + low) / 2
                    val = self.cdf(parms, new)
                    if val > y:
                        up = new
                    else: 
                        low = new
                return(new)
        
        
        self.ppf = ppf
    

        


