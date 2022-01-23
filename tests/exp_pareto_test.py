import bayes_splicing as bs
import pandas as pd
import numpy as np

# Definition of the exponential-Pareto composite model
f1, f2 = bs.loss_model("Exp", ["λ1"]), bs.loss_model("Pareto-Tail", ["α2"])

# Type of composite models include "disjoint", "simple" and "continuous"
model_type = "continuous"
f = bs.spliced_loss_model(f1, f2, model_type)

# We will fit the above model to the famous danish fire insurance data set
# danish = pd.read_csv("data/danish.csv").x
# X = danish.values
parms_true = np.array([1 / 3, 2, 1])
X = f.sample(parms_true, 1000)

# Prior distributions over the model parameters
λ1_prior = bs.prior_model('gamma','λ1', 1, 1)
α2_prior = bs.prior_model('gamma','α2',1, 1)
γ_prior =  bs.prior_model('uniform','γ',min(X), max(X))
prior_spliced_model = bs.independent_priors([λ1_prior,α2_prior, γ_prior])


# The smc function deliver the trace, the log marginal likelihood along with some information criteria
popSize = 2000
trace, log_marg, DIC, WAIC = bs.smc(X, f, popSize, prior_spliced_model)

# The posterior distributions
bs.posterior_plots(f, trace)
# The trace plot
bs.trace_plots(f, trace)
# The quantile-quantile plot
f.set_ppf()
bs.qq_plot(X, f, trace.mean().values)

# Computation of the pure premium and the XOL premium with priority P and limit L

f.set_pdf()
# Based on the mean a posteriori
parms = trace.mean().values 
f.PP(parms)
P, L = 2, 6
f.XOLP( parms, P, L)

# Based on the whole posterior sample
pps = [f.PP(parms) for parms in trace.values]
xolps = [f.XOLP(parms, P, L) for parms in trace.values]
np.mean(pps), np.mean(xolps)

# Based on the mean a posteriori
parms = trace.mean().values 
expo, premiums, safety_loadings, n_sim = 100, np.array([f.PP(parms), f.XOLP(parms, P, L)]), [0.05, 0.05], 1e5
pnls = f.PnL(parms, P, L, expo, premiums, safety_loadings, n_sim)
np.quantile(pnls, 0.005), np.mean(pnls)

# Based on the whole posterior sample
expo, premiums, safety_loadings, n_sim = 100, np.array([np.mean(pps), np.mean(xolps)]), [0.05, 0.05], 1e5
pnls = [f.PnL(parms, P, L, expo, premiums) for parms in trace.sample(int(n_sim), replace = True).values]
np.quantile(pnls, 0.005), np.mean(pnls)
