# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:04:02 2021

@author: pierr
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

def post_PnL(fs, ws, traces, expo, P = 0, L = np.inf, safety_loadings = [0.05, 0.05], n_sim = 1):
    [f.set_pdf() for f in fs]
    model_indices = np.random.choice(list(range(len(fs))), int(n_sim), ws)
    PnLs = []
    for k in model_indices:
        parms = traces[k].sample().values[0]
        PnLs += fs[k].PnL(parms, expo, P, L, safety_loadings, n_sim = 1)
    return(PnLs)

def posterior_plots(f, trace):
    fig, axs = plt.subplots(1, f.d, figsize=(10, 3.5))
    x_labs = trace.columns
    for k in range(len(x_labs)):
        if len(x_labs) > 1:
            if trace[x_labs[k]].var() > 0:
                positions = np.linspace(min(trace[x_labs[k]]), max(trace[x_labs[k]]), 1000)
                kernel = st.gaussian_kde(trace[x_labs[k]], bw_method = 1)
                axs[k].plot(positions, kernel(positions), lw=3, label = "a posteriori", color = "blue")
                axs[k].axvline(x=trace[x_labs[k]].mean(), label = "MAP", color='black', linestyle='dashed' )
                axs[k].set_xticks(np.round(
                trace[x_labs[k]].quantile([0.001, 0.999]).values, 2))
            # else:
                # axs[k].axvline(x=trace[x_labs[k]].mean())
                
            axs[k].set_yticks([])
            
            axs[k].set_xlabel(x_labs[k])
            handles, labels = axs[0].get_legend_handles_labels()
        else:
            positions = np.linspace(min(trace[x_labs[k]]), max(trace[x_labs[k]]), 1000)
            kernel = st.gaussian_kde(trace[x_labs[k]], bw_method = 1)
            axs.plot(positions, kernel(positions), lw=3, label = "a posteriori", color = "blue")
            axs.axvline(x=trace[x_labs[k]].mean(), label = "MAP", color='black', linestyle='dashed' )
            axs.set_yticks([])
            axs.set_xticks(np.round(
                trace[x_labs[k]].quantile([0.001, 0.999]).values, 2))
            axs.set_xlabel(x_labs[k])
            handles, labels = axs.get_legend_handles_labels()
    
    
    fig.legend(handles, labels, ncol = 3, borderaxespad=0., loc='upper center', 
               frameon=False)
    sns.despine()
    fig.tight_layout()

def trace_plots(f, trace):
    fig, axs = plt.subplots(1, f.d, figsize=(10, 3.5))
    x_labs = f.parm_names
    for k in range(len(x_labs)):
        if f.d > 1:
            axs[k].plot(trace[trace.columns.values[k]], lw=3, label = "a posteriori", color = '#006699')
            axs[k].set_xticks([])
            axs[k].set_yticks(np.round(
                trace[f.parm_names[k]].quantile([0.001, 0.999]).values, 2))
            axs[k].set_xlabel(x_labs[k])
            handles, labels = axs[0].get_legend_handles_labels()
        else:
            axs.plot(trace[trace.columns.values[k]], lw=3, label = "a posteriori", color = '#006699')
            axs.set_xticks([])
            axs.set_yticks(np.round(
                trace[f.parm_names[k]].quantile([0.001, 0.999]).values, 2))
            axs.set_xlabel(x_labs[k])
            handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, ncol = 3, borderaxespad=0., loc='upper center', 
               frameon=False)
    sns.despine()
    fig.tight_layout()
    
def qq_plot(X, f, parms):
    y_vec = np.arange(1, len(X), 1)/len(X)
    fig, axs = plt.subplots(1, 1, figsize=(2.5, 2.5))
    plt.plot(np.sort(X[:(len(X)-1)]),np.array([f.ppf(parms, y) for y in y_vec]), "ob", color = '#006699')
    plt.plot(np.sort(X[:(len(X)-1)]),np.sort(X[:(len(X)-1)]), ":k", lw=2)
    plt.xlabel("Empirical")
    plt.ylabel("Theoretical")
    sns.despine()
    
def compute_Wasserstein(X, f, parms, p):
    y_vec = np.arange(1, len(X), 1)/len(X)
    return(np.sum(abs(np.sort(X[:(len(X)-1)]) -np.array([f.ppf(parms, y) for y in y_vec]))**p)**(1/p))

