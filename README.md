# _bayes-splicing_ Python package for Bayesian inference of splicing distributions

This package implements an efficient Sequential Monte Carlo Sampling (SMC) algorithm. Splicing models, also called composite models, are fitted to the data and compared using Bayesian statistics tools

This package is the result of my paper "[Sequential Monte Carlo Sampler to fit and compare insurance loss models](https://hal.archives-ouvertes.fr/hal-03263471v1)". To install simply run `pip install bayes-splicing`.

For example, imagine we have a sample of insurance losses and we would like to fit a composite model based on the exponential and Pareto distributions. 

```python
import bayes_splicing as bs
import pandas as pd
import numpy as np

# Definition of the exponential-Pareto composite model
f1, f2 = bs.loss_model("Exp", ["lambda"]), bs.loss_model("Pareto-Tail", ["alpha"])

# Type of composite models include "disjoint", "simple" and "continuous"
model_type = "continuous"
f = bs.spliced_loss_model(f1, f2, model_type, thresh_name="gamma")

# We will fit the above model to the famous danish fire insurance data set
# danish = pd.read_csv("data/danish.csv").x
# X = danish.values
parms_true = np.array([1 / 3, 2, 1])
X = f.sample(parms_true, 1000)

# Prior distributions over the model parameters
lambda_prior = bs.prior_model('gamma','lambda', 1, 1)
alpha_prior = bs.prior_model('gamma','alpha',1, 1)
gamma_prior =  bs.prior_model('uniform','gamma',min(X), max(X))
prior_spliced_model = bs.independent_priors([lambda_prior,alpha_prior, gamma_prior])


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
```