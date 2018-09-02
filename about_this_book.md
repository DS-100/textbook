---
prev_page: '/'
next_page: '/ch/01/lifecycle_intro.html'
---
## About This Book

In this book, we will proceed as though the reader is comfortable with the
knowledge presented in [Data 8][data8] or some equivalent. In particular, we
will assume that the reader is familiar with the following topics (links to
pages from the Data 8 textbook are given in parentheses).

* Tabular data manipulation: selection, filtering, grouping, joining [(link)][8.2]
* Basic probability concepts [(link)][9.5]
* Sampling, empirical distributions of statistics [(link)][10.3]
* Hypothesis testing using bootstrap resampling [(link)][13.4]
* Least squares regression and regression inference [(link)][16.2]
* Classification [(link)][17.1]

In addition, we assume that the reader has taken a course in computer
programming in Python, such as [CS61A][61a] or some equivalent. We will not
explain Python syntax except in special cases.

Finally, we assume that the reader has basic familiarity with partial
derivatives, gradients, vector algebra, and matrix algebra.

### Notation

This book covers topics from multiple disciplines. Unfortunately, some of these
disciplines use the same notation to describe different concepts. In order to
prevent headaches, we have devised notation that may differ slightly
from the notation used in your discipline.

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
$ \boldsymbol{\theta^\*} = [ \theta^\*_1, \theta^\*_2, \ldots, \theta^\*_n ] $
and a vector of fitted model parameters as
$ \boldsymbol{\hat{\theta}} = [\hat{\theta_1}, \hat{\theta_2}, \ldots, \hat{\theta_n} ] $.

We will always use bold uppercase letters for matrices. For example, we
commonly represent a data matrix using $ \boldsymbol X $.

We will always use non-bolded uppercase letters for random variables, such as
$ X $ or $ Y $.

When discussing the bootstrap, we use $ \theta^* $ to denote the population
parameter, $ \hat{\theta} $ to denote the sample test statistic, and
$ \tilde{\theta} $ to denote a bootstrapped test statistic.

[8.2]: https://www.inferentialthinking.com/chapters/08/2/classifying-by-one-variable.html
[9.5]: https://www.inferentialthinking.com/chapters/09/5/finding-probabilities.html
[10.3]: https://www.inferentialthinking.com/chapters/10/3/empirical-distribution-of-a-statistic.html
[13.4]: https://www.inferentialthinking.com/chapters/13/4/using-confidence-intervals.html
[16.2]: https://www.inferentialthinking.com/chapters/16/2/inference-for-the-true-slope.html
[17.1]: https://www.inferentialthinking.com/chapters/17/1/nearest-neighbors.html
[data8]: http://data8.org/
[61a]: https://cs61a.org/
