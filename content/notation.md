# Notation

This book covers topics from multiple disciplines. Unfortunately, some of these
disciplines use the same notation to describe different concepts. For clarity,
we have devised notation that may differ slightly from the notation used in
your discipline.

A population parameter is denoted by $ \theta^* $. The model parameter that
minimizes a specified loss function is denoted by $ \hat{\theta} $. Typically,
we desire $ \hat{\theta} \approx \theta^* $. We use the plain variable
$ \theta $ to denote a model parameter that does not minimize a particular loss
function. For example, we may arbitrarily set $ \theta = 16$ in order to
calculate a model's loss at that choice of $ \theta $. When using gradient
descent to minimize a loss function, we use $ \theta^{(t)} $ to represent the
intermediate values of $ \theta $.

We will always use bold lowercase letters for vectors. For example, we
represent a vector of population parameters using
$ \boldsymbol{\theta^*} = [ \theta^*_1, \theta^*_2, \ldots, \theta^*_n ] $
and a vector of fitted model parameters as
$ \boldsymbol{\hat{\theta}} = [\hat{\theta_1}, \hat{\theta_2}, \ldots, \hat{\theta_n} ] $.

We will always use bold uppercase letters for matrices. For example, we
commonly represent a data matrix using $ \boldsymbol X $.

We will always use non-bolded uppercase letters for random variables, such as
$ X $ or $ Y $.

When discussing the bootstrap, we use $ \theta^* $ to denote the population
parameter, $ \hat{\theta} $ to denote the sample test statistic, and
$ \tilde{\theta} $ to denote a bootstrapped test statistic.
