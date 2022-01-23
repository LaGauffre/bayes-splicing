# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:22:08 2020

@author: pierr
"""
import numpy as np
import numba as nb
import math as ma

class prior_model:
    def __init__(self, name, parm_name, a , b):
        self.name = name
        self.parm_name = parm_name
        self.a = a
        self.b = b
    
    def set_logp(self):
        a, b = self.a, self.b
        if self.name == "gamma":
            def logp(x):
                if x > 0:
                    return(( a - 1) * np.log(x) - b * x + a * np.log(b) - np.log(ma.gamma(a)))
                else:
                    return(-np.inf)
        elif self.name == "beta":
            def logp(x):
                if x>0:
                    return((a - 1)*np.log(x) + (b - 1)*np.log(1 - x) -  np.log(ma.gamma(a)) - np.log(ma.gamma(b)) + np.log(ma.gamma(a + b)))
                else:
                    return(-np.inf)
        elif self.name == "uniform":
            def logp(x):
                if x > a and x < b:
                    return(-np.log(b-a))
                else:
                    return(-np.inf)
        elif self.name == "normal":
            def logp(x):
                return(- np.log(2 * ma.pi) / 2 - (x-a)**2 / 2 / b**2 )
        elif self.name == "fixed":
            def logp(x):
                if x == a:
                    return(0)
                else:
                    return(-np.inf)
        self.logp = nb.jit(nopython = True)(logp)
    
    def sample(self, n):
        if self.name == "gamma":
            return(np.random.gamma(self.a, 1 / self.b, size = n))
        elif self.name == "beta":
            return(np.random.beta(self.a, self.b, size = n))
        elif self.name == "uniform":
            return(np.random.uniform(low = self.a, high = self.b, size = n))
        elif self.name == "normal":
            return(np.random.normal(self.a, self.b, size = n))
        elif self.name == "fixed":
            return(self.a * np.ones(n))


    
    
        
