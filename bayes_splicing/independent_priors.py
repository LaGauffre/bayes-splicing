# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:17:31 2021

@author: pierr
"""
import numpy as np
import pandas as pd
import numba as nb

class independent_priors:
    def __init__(self, priors):
        self.priors = priors
        self.parm_names = [prior.parm_name for prior in priors]
        self.d = len(priors)
        self.name = '-'.join([prior.name for prior in priors])
    
    def sample(self,n):
        particle_cloud = pd.DataFrame(
            np.array([prior.sample(n) for prior in self.priors]).T)
        particle_cloud.columns = self.parm_names
        return(particle_cloud)
    
    def set_logp(self):
        [prior.set_logp() for prior in self.priors]
        if self.d == 1:
            logps_prior0 = self.priors[0].logp 
            def logp(parms):
                return(logps_prior0(parms[0]))
        elif self.d == 2:
            logps_prior0, logps_prior1 = tuple([prior.logp for prior in self.priors])
            def logp(parms):
                return(logps_prior0(parms[0]) + logps_prior1(parms[1]))
        elif self.d == 3:
            logps_prior0, logps_prior1, logps_prior2 \
                = tuple([prior.logp for prior in self.priors])
            def logp(parms):
                return(logps_prior0(parms[0]) + logps_prior1(parms[1]) + 
                        logps_prior2(parms[2]))
        elif self.d == 4:
            logps_prior0, logps_prior1, logps_prior2, logps_prior3,\
                = tuple([prior.logp for prior in self.priors])
            def logp(parms):
                return(logps_prior0(parms[0]) + logps_prior1(parms[1]) + 
                        logps_prior2(parms[2]) + logps_prior3(parms[3]))
        elif self.d == 5:
            logps_prior0, logps_prior1, logps_prior2, logps_prior3, logps_prior4,\
                = tuple([prior.logp for prior in self.priors])
            def logp(parms):
                return(logps_prior0(parms[0]) + logps_prior1(parms[1]) + 
                        logps_prior2(parms[2]) + logps_prior3(parms[3]) + 
                        logps_prior4(parms[4]))
        elif self.d == 6:
            logps_prior0, logps_prior1, logps_prior2, logps_prior3, logps_prior4,\
                logps_prior5 = tuple([prior.logp for prior in self.priors])
            def logp(parms):
                return(logps_prior0(parms[0]) + logps_prior1(parms[1]) + 
                        logps_prior2(parms[2]) + logps_prior3(parms[3]) + 
                        logps_prior4(parms[4]) + logps_prior5(parms[5]))
        elif self.d == 7:
            logps_prior0, logps_prior1, logps_prior2, logps_prior3, logps_prior4,\
                logps_prior5, logps_prior6 = tuple([prior.logp for prior in self.priors])
            def logp(parms):
                return(logps_prior0(parms[0]) + logps_prior1(parms[1]) + 
                        logps_prior2(parms[2]) + logps_prior3(parms[3]) + 
                        logps_prior4(parms[4]) + logps_prior5(parms[5]) +
                        logps_prior6(parms[6]))
        elif self.d == 8:
            logps_prior0, logps_prior1, logps_prior2, logps_prior3, logps_prior4,\
                logps_prior5, logps_prior6, logps_prior7 \
                    = tuple([prior.logp for prior in self.priors])
            def logp(parms):
                return(logps_prior0(parms[0]) + logps_prior1(parms[1]) + 
                        logps_prior2(parms[2]) + logps_prior3(parms[3]) + 
                        logps_prior4(parms[4]) + logps_prior5(parms[5]) +
                        logps_prior6(parms[6]) + logps_prior7(parms[7]))

        
          
        self.logp = nb.jit(nopython = True)(logp)
    