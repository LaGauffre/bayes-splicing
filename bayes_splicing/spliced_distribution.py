# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:21:11 2021

@author: pierr
"""
import numpy as np
import numba as nb
import scipy as sc


class spliced_loss_model():
    
    def __init__(self, f1, f2, model_type, thresh_name = 'γ', mixing_name = 'p' ):
        self.body = f1
        self.tail = f2
        self.model_type = model_type
        if model_type == "disjoint":
            self.d = f1.d + f2.d + 2 
            self.parm_names = np.append(np.append(f1.parm_names, f2.parm_names), 
                                        np.array([thresh_name, mixing_name]))
        elif model_type == "simple" or  model_type == "continuous":
            self.d = f1.d + f2.d + 1
            self.parm_names = np.append(np.append(f1.parm_names, f2.parm_names), 
                                        np.array([thresh_name]))
        self.body.set_logps(), self.tail.set_logps(), self.body.set_cdf(), self.tail.set_cdf(), self.body.set_pdf(), self.tail.set_pdf(), self.body.set_logds(), self.tail.set_logds(), self.body.set_ppf(), self.tail.set_ppf()

    
    def set_logp(self, X):
        
        body_d, tail_d, logp_body, logp_tail = self.body.d, self.tail.d,\
            self.body.logp_body,self.tail.logp_tail
        
        model_type, cdf1, cdf2, pdf1, pdf2, f2_name = self.model_type, self.body.cdf, self.tail.cdf, self.body.pdf, self.tail.pdf, self.tail.name
        def logp(parms):
            parms_f1, parms_f2, γ = parms[:body_d], \
                parms[body_d: (body_d + tail_d)], \
                    parms[(body_d + tail_d)]
            if "Tail" in f2_name:
                if model_type == "disjoint":
                    p = parms[-1]
                elif model_type == "simple":
                    S1, S2 = cdf1(parms_f1, γ), cdf2(parms_f2, γ, γ)
                    s1, s2 = pdf1(parms_f1, γ), pdf2(parms_f2, γ, γ)
                    if S1 > 0 and S2 < 1 and s1 > 0 and s2 > 0 :
                        p =  S1
                    else:
                        p = 0

                elif model_type == "continuous":
                    S1, S2 = cdf1(parms_f1, γ), cdf2(parms_f2, γ, γ)
                    s1, s2 = pdf1(parms_f1, γ), pdf2(parms_f2, γ, γ)
                    if S1 > 0 and S2 < 1 and s1 > 0 and s2 > 0 :
                        p =  s2 / (1 - S2) / ( s2 / (1 - S2) + s1 / S1)

                    else:
                        p = 0
            else:
                if model_type == "disjoint":
                    p = parms[-1]
                elif model_type == "simple":
                    S1, S2 = cdf1(parms_f1, γ), cdf2(parms_f2, γ)
                    s1, s2 = pdf1(parms_f1, γ), pdf2(parms_f2, γ)
                    if S1 > 0 and S2 < 1 and s1 > 0 and s2 > 0 :
                        p =  S1
                    else:
                        p = 0

                elif model_type == "continuous":
                    S1, S2 = cdf1(parms_f1, γ), cdf2(parms_f2, γ)
                    s1, s2 = pdf1(parms_f1, γ), pdf2(parms_f2, γ)
                    if S1 > 0 and S2 < 1 and s1 > 0 and s2 > 0 :
                        p =  s2 / (1 - S2) / ( s2 / (1 - S2) + s1 / S1)

                    else:
                        p = 0
            
            if γ > 0 and p > 0 and p < 1:
                X1 = X[X < γ]
                X2 = X[X >= γ]
                return(len(X1) * np.log(p) + 
                       logp_body(X1, parms_f1, γ) + 
                       len(X2) * np.log(1-p) + logp_tail(X2, parms_f2, γ))
            else:
                return(-np.inf)
        self.logp = nb.jit(nopython = True)(logp)
        
    def sample(self, parms, n):
        parms_f1, parms_f2, γ = parms[:self.body.d],\
            parms[self.body.d: (self.body.d + self.tail.d)], \
                parms[(self.body.d + self.tail.d)]
        model_type, cdf1, cdf2, pdf1, pdf2, f2_name = self.model_type, self.body.cdf, self.tail.cdf, self.body.pdf, self.tail.pdf, self.tail.name
        if "Tail" in f2_name:
                if model_type == "disjoint":
                    p = parms[-1]
                elif model_type == "simple":
                    S1, S2 = cdf1(parms_f1, γ), cdf2(parms_f2, γ, γ)
                    s1, s2 = pdf1(parms_f1, γ), pdf2(parms_f2, γ, γ)
                    if S1 > 0 and S2 < 1 and s1 > 0 and s2 > 0 :
                        p =  S1
                    else:
                        p = 0

                elif model_type == "continuous":
                    S1, S2 = cdf1(parms_f1, γ), cdf2(parms_f2, γ, γ)
                    s1, s2 = pdf1(parms_f1, γ), pdf2(parms_f2, γ, γ)
                    if S1 > 0 and S2 < 1 and s1 > 0 and s2 > 0 :
                        p =  s2 / (1 - S2) / ( s2 / (1 - S2) + s1 / S1)

                    else:
                        p = 0
        else:
            if model_type == "disjoint":
                p = parms[-1]
            elif model_type == "simple":
                S1, S2 = cdf1(parms_f1, γ), cdf2(parms_f2, γ)
                s1, s2 = pdf1(parms_f1, γ), pdf2(parms_f2, γ)
                if S1 > 0 and S2 < 1 and s1 > 0 and s2 > 0 :
                    p =  S1
                else:
                    p = 0

            elif model_type == "continuous":
                S1, S2 = cdf1(parms_f1, γ), cdf2(parms_f2, γ)
                s1, s2 = pdf1(parms_f1, γ), pdf2(parms_f2, γ)
                if S1 > 0 and S2 < 1 and s1 > 0 and s2 > 0 :
                    p =  s2 / (1 - S2) / ( s2 / (1 - S2) + s1 / S1)

                else:
                    p = 0        
        ber_rvs = np.random.binomial(1, p, size=n)
        unif_rvs = np.random.uniform(size = n)
        if "Tail" in f2_name:
            body_rvs = self.body.ppf(parms_f1, 
                                     self.body.cdf(parms_f1, γ) * unif_rvs )
            tail_rvs = self.tail.ppf(parms_f2, γ,
                                     1 - (1 - self.tail.cdf(parms_f2, γ, γ)) * unif_rvs)
        else:
            body_rvs = self.body.ppf(parms_f1, 
                                     self.body.cdf(parms_f1, γ) * unif_rvs )
            tail_rvs = self.tail.ppf(parms_f2,
                                     1 - (1 - self.tail.cdf(parms_f2, γ)) * unif_rvs)
            
        
        return(ber_rvs * body_rvs + (1 - ber_rvs) * tail_rvs)
    
    def set_logd(self, parms):
        parms_f1, parms_f2, γ = parms[:,:self.body.d],\
            parms[:,self.body.d: (self.body.d + self.tail.d)],\
                parms[:,(self.body.d + self.tail.d)] 
        model_type, cdf1, cdf2, pdf1, pdf2, f2_name = self.model_type, self.body.cdf, self.tail.cdf, self.body.pdf, self.tail.pdf, self.tail.name
        
        if model_type == "disjoint":
            p = parms[:,-1]
        elif model_type == "simple":
            p = np.array([cdf1(parm_f1, γ[j] ) for j, parm_f1 in enumerate(parms_f1)])
        elif model_type == "continuous":
            if "Tail" in f2_name:
                p = np.array([pdf2(parm_f2, gam, gam) / (1 - cdf2(parm_f2, gam, gam)) / ( pdf2(parm_f2, gam, gam) / (1 - cdf2(parm_f2, gam, gam)) + pdf1(parm_f1, gam) / cdf1(parm_f1, gam)) for parm_f1, parm_f2, gam in zip(parms_f1, parms_f2, γ) ])
            else:
                p = np.array([pdf2(parm_f2, gam) / (1 - cdf2(parm_f2, gam)) / ( pdf2(parm_f2, gam) / (1 - cdf2(parm_f2, gam)) + pdf1(parm_f1, gam) / cdf1(parm_f1, gam)) for parm_f1, parm_f2, gam in zip(parms_f1, parms_f2, γ) ])
        def logd(x):
            res = np.zeros(len(γ))
            s = np.logical_and(p > 0, p < 1)
            s1 = np.logical_and(s, x < γ)
            s2 = np.logical_and(s, x >= γ)
            res1 = self.body.logd_body(x, parms_f1[s1], γ[s1]) + np.log(p[s1])    
            res2 = self.tail.logd_tail(x, parms_f2[s2], γ[s2]) + np.log(1 - p[s2])
            res[np.where(s1)] = res1
            res[np.where(s2)] = res2
            res[np.where(np.invert(s))] = - np.inf
            return(res)
        self.logd = logd
    
    def set_cdf(self):
        d_body, d_tail = self.body.d, self.tail.d
        model_type, cdf1, cdf2, pdf1, pdf2, f2_name = self.model_type, self.body.cdf, self.tail.cdf, self.body.pdf, self.tail.pdf, self.tail.name
        def cdf(parms, x):
            parms_f1, parms_f2, γ = parms[:d_body],\
                parms[d_body: (d_body + d_tail)], \
                    parms[(d_body + d_tail)]
            if model_type == "disjoint":
                p = parms[-1]
            elif model_type == "simple":
                p = cdf1(parms_f1, γ)
            elif model_type == "continuous":
                if "Tail" in f2_name:
                    p =  pdf2(parms_f2, γ, γ) / (1 - cdf2(parms_f2, γ, γ)) / ( pdf2(parms_f2, γ, γ) / (1 - cdf2(parms_f2, γ, γ)) + pdf1(parms_f1, γ) / cdf1(parms_f1, γ)) 
                else:
                    p =  pdf2(parms_f2, γ) / (1 - cdf2(parms_f2, γ)) / ( pdf2(parms_f2, γ) / (1 - cdf2(parms_f2, γ)) + pdf1(parms_f1, γ) / cdf1(parms_f1, γ)) 
            
            
            if x < γ:
                return(p * cdf1(parms_f1, x) / cdf1(parms_f1, γ))
            else:
                if "Tail" in f2_name:
                    return(p + (1-p) * (cdf2(parms_f2, γ,  x)- cdf2(parms_f2, γ,  γ)) / (1 - cdf2(parms_f2, γ, γ)) )
                else:
                    return(p + (1-p) * (cdf2(parms_f2, x)- cdf2(parms_f2, γ)) / (1 - cdf2(parms_f2, γ)) )
        self.cdf = nb.jit(nopython = True)(cdf)
    
    def set_pdf(self):
        d_body, d_tail = self.body.d, self.tail.d
        model_type, cdf1, cdf2, pdf1, pdf2, f2_name = self.model_type, self.body.cdf, self.tail.cdf, self.body.pdf, self.tail.pdf, self.tail.name
        def pdf(parms, x):
            parms_f1, parms_f2, γ = parms[:d_body],\
                parms[d_body: (d_body + d_tail)], \
                    parms[(d_body + d_tail)]
            if model_type == "disjoint":
                p = parms[-1]
            elif model_type == "simple":
                p = cdf1(parms_f1, γ)
            elif model_type == "continuous":
                if "Tail" in f2_name:
                    p =  pdf2(parms_f2, γ, γ) / (1 - cdf2(parms_f2, γ, γ)) / ( pdf2(parms_f2, γ, γ) / (1 - cdf2(parms_f2, γ, γ)) + pdf1(parms_f1, γ) / cdf1(parms_f1, γ)) 
                else:
                    p =  pdf2(parms_f2, γ) / (1 - cdf2(parms_f2, γ)) / ( pdf2(parms_f2, γ) / (1 - cdf2(parms_f2, γ)) + pdf1(parms_f1, γ) / cdf1(parms_f1, γ)) 
            
            
            if x < γ:
                return(p * pdf1(parms_f1, x) / cdf1(parms_f1, γ))
            else:
                if "Tail" in f2_name:
                    return((1-p) *pdf2(parms_f2, γ,  x) / (1 - cdf2(parms_f2, γ, γ)) )
                else:
                    return((1-p) * pdf2(parms_f2, x) / (1 - cdf2(parms_f2, γ)))
        self.pdf = nb.jit(nopython = True)(pdf)
        
    def set_ppf(self):
        self.set_cdf()
        cdf = self.cdf
        def ppf(parms, y):
            low, up = 0, 1e8
            err = 1e-6
            while up - low > err:
                new = (up + low) / 2
                val = cdf(parms, new)
                if val > y:
                    up = new
                else: 
                    low = new
            return(new)
        self.ppf = nb.jit(nopython = True)(ppf)
    
    def PP(self, parms):
        return(sc.integrate.quad(lambda x: x*self.pdf(parms, x), 0, np.inf)[0])
    
    def XOLP(self, parms, P, L):
        return(sc.integrate.quad(lambda x: min(x - P, L) * self.pdf(parms, x), P, np.inf)[0])
    
    def PnL(self, parms, P, L, expo, premiums, safety_loadings = [0.05, 0.05], n_sim = 1):
        π, π2 = premiums
        π1 = (1 + safety_loadings[0]) * π - (1 + safety_loadings[1]) * π2
        PnLs = []
        for k in range(int(n_sim)):
            N = np.random.poisson(expo, 1)
            X = self.sample(parms, N)
            S = np.sum(X)
            if L == np.inf:
                R = np.sum(X[X > P] - P) 
            else:
                R = np.sum(X[np.logical_and(X > P, X - P < L)] - P) + \
                    np.sum(np.logical_and(X > P, X - P > L)) * L
            D = S - R
            PnLs.append(π1 * expo - D)
        return(PnLs)

    