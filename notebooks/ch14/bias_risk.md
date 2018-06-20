
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Risk-and-Cost-Minimization" data-toc-modified-id="Risk-and-Cost-Minimization-1">Risk and Cost Minimization</a></span></li><li><span><a href="#Risk" data-toc-modified-id="Risk-2">Risk</a></span></li><li><span><a href="#Empirical-Risk" data-toc-modified-id="Empirical-Risk-3">Empirical Risk</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-4">Summary</a></span></li></ul></div>


```python
# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
```


```python
# HIDDEN
def df_interact(df, nrows=7, ncols=7):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0, col=0):
        return df.iloc[row:row + nrows, col:col + ncols]
    if len(df.columns) <= ncols:
        interact(peek, row=(0, len(df) - nrows, nrows), col=fixed(0))
    else:
        interact(peek,
                 row=(0, len(df) - nrows, nrows),
                 col=(0, len(df.columns) - ncols))
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))
```

## Risk and Cost Minimization

In order to make predictions using data, we define a model, select a cost function, and fit the model's parameters by minimizing the cost. For example, to conduct least squares linear regression, we select the model:

$$
\begin{aligned}
f_\hat{\theta} (x) &= \hat{\theta} \cdot x
\end{aligned}
$$

And the cost function:

$$
\begin{aligned}
L(\hat{\theta}, X, y)
&= \frac{1}{n} \sum_{i}(y_i - f_\hat{\theta} (X_i))^2\\
\end{aligned}
$$

As before, we use $ \hat{\theta} $ as our vector of model parameters, $ x $ as a vector containing a row of a data matrix $ X $, and $ y $ as our vector of observed values to predict. $ X_i $is the $i$'th row of $ X $ and $ y_i $ is the $i$'th entry of y.

Observe that our cost function is the average of the loss function values for each row of our data. If we define the squared loss function:

$$
\begin{aligned}
\ell(y_i, f_\hat{\theta} (x))
&= (y_i - f_\hat{\theta} (x))^2
\end{aligned}
$$

Then we may rewrite our cost function more simply:

$$
\begin{aligned}
L(\hat{\theta}, X, y)
&= \frac{1}{n} \sum_{i} \ell(y_i, f_\hat{\theta} (X_i))
\end{aligned}
$$

The expression above abstracts over the specific loss function; regardless of the loss function we choose, our cost is the average loss.

By minimizing the cost, we select the model parameters that best fit our observed dataset. Thus far, we have refrained from making statements about the population that generated the dataset. In reality, however, we are quite interested in making good predictions on the entire population, not just our data that we have already seen.

## Risk

If our observed dataset $ X $ and $ y $ are drawn at random from a given population, our observed data are random variables. If our observed data are random variables, our model parameters are also random variables—each time we collect a new set of data and fit a model, the parameters of the model $ f_\hat{\theta} (x) $ will be slightly different.

Suppose we draw one more input-output pair $z, \gamma $ from our population at random. The loss that our model produces on this value is:

$$
\begin{aligned}
\ell(\gamma, f_\hat{\theta} (z))
\end{aligned}
$$

Notice that this loss is a random variable; the loss changes for different sets of observed data $ X $ and $ y $ and different points $z, \gamma $ from our population.

The **risk** for a model $ f_\hat{\theta} $ is the expected value of the loss above for all training data $ X $, $ y $ and all points $ z$, $ \gamma $ in the population:

$$
\begin{aligned}
R(f_\hat{\theta}(x)) = \mathbb{E}[ \ell(\gamma, f_\hat{\theta} (z)) ]
\end{aligned}
$$

Notice that the risk is an expectation of a random variable and is thus *not* random itself. The expected value of fair six-sided die rolls is 3.5 even though the rolls themselves are random.

The risk above is sometimes called the **true risk** because it tells how a model does on the entire population. If we could compute the true risk for all models, we can simply pick the model with the least risk and know with certainty that the model will perform better in the long run than all other models on our choice of loss function.

## Empirical Risk

Reality, however, is not so kind. If we substitute in the definition of expectation into the formula for the true risk, we get:

$$
\begin{aligned}
R(f_\hat{\theta})
&= \mathbb{E}[ \ell(\gamma, f_\hat{\theta} (z)) ] \\
&= \sum_\gamma \sum_z \ell(\gamma, f_\hat{\theta} (z)) P(\gamma, z) \\
\end{aligned}
$$

To further simplify this expression, we need to know $ P(\gamma, z)  $, the global probability distribution of observing any point in the population. Unfortunately, this is not so easy. Suppose we are trying to predict the tip amount based on the size of the table. What is the probability that a table of three people gives a tip of $14.50? If we knew the distribution of points exactly, we wouldn't have to collect data or fit a model—we would already know the most likely tip amount for any given table.

Although we do not know the exact distribution of the population, we can approximate it using the observed dataset $ X $ and $ y $. If $ X $ and $ y $ are drawn at random from our population, the distribution of points in $ X $ and $ y $ is similar to the population distribution. Thus, we treat $ X $ and $ y $ as our population. Then, the probability that any input-output pair $ X_i $, $ y_i $ appear is $ \frac{1}{n} $ since each pair appears once out of $ n $ points total.

This allows us to calculate the **empirical risk**, an approximation for the true risk:

$$
\begin{aligned}
\hat R(f_\hat{\theta})
&= \mathbb{E}[ \ell(y_i, f_\hat{\theta} (X_i)) ] \\
&= \sum_{i=1}^n \ell(y_i, f_\hat{\theta} (X_i)) \frac{1}{n} \\
&= \frac{1}{n} \sum_{i=1}^n \ell(y_i, f_\hat{\theta} (X_i)) 
\end{aligned}
$$

If our dataset is large and the data are drawn at random from the population, the empirical risk $ \hat R(f_\hat{\theta}) $ is close to the true risk $ R(f_\hat{\theta}) $. This allows us to pick the model that minimizes the empirical risk.

Notice that this expression is the cost function at the start of the section! By minimizing the average loss, we also minimize the empirical risk. This explains why we often use the average loss as our cost function instead of the maximum loss, for example.

## Summary

The true risk of a prediction model describes the overall long-run loss that the model will produce for the population. Since we typically cannot calculate the true risk directly, we calculate the empirical risk instead and use the empirical risk to find an appropriate model for prediction. Because the empirical risk is the average loss on the observed dataset, we often minimize the average loss when fitting models.
