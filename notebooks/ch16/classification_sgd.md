

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
pd.set_option('precision', 2)
```

# Limitations of Gradient Descent

In chapter 11, we covered the gradient descent algorithm, which is used to converge at the $\theta$ parameters that minimize a loss function $L(\theta, y)$.

$$
\theta_{t+1} = \theta_t - \alpha \cdot \nabla_\theta L(\theta, y)
$$

We can replace  $\nabla_\theta L(\theta, y)$ with the gradient of the cross entropy cost that we previously derived to find the gradient descent algorithm specific to logistic regression. Letting $ \sigma_i = f_\hat{\theta}(X_i) = \sigma(X_i \cdot \hat \theta) $, the gradient descent algorithm becomes:

$$
\theta_{t+1} = \theta_t - \alpha \cdot \left(- \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sigma_i\right) X_i \right)
$$

By definition, the gradient of the cross entropy cost is the average of the gradient of the loss over all $n$ observations. This is then computed at each iteration $t$ of the gradient descent algorithm. For large $n$, this can become a computationally expensive algorithm.

# Stochastic Gradient Descent

An alternative to gradient descent is **stochastic gradient descent**, in which we approximate the gradient by taking a sample of observations. The gradient of the loss function in stochastic gradient descent is below, with $ \sigma_i = f_\hat{\theta}(X_i) = \sigma(X_i \cdot \hat \theta) $.

$$
\nabla_\theta L(\theta, y) \approx -\frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}(y_i - \sigma_i)X_i
$$

$\mathcal{B}$ is a batch of data points that we randomly sample from the $n$ observations. At each iteration of stochastic gradient descent, we will make an estimate of the true gradient of the loss function using a new randomly sampled batch. Because $\mathcal{B}$ is a simple random sample, the expectation of the gradient of the loss function over the batch is equal to the true gradient over all $n$ observations. In other words, on average, we expect that stochastic gradient descent will converge to the $\theta$ parameters that minimize the loss function.

## Selecting the batch size

Stochastic gradient descent generally refers to an algorithm with a batch size of 1, and mini-batch gradient descent is often used to describe algorithms with a randomly sampled batch of size smaller than $n$. In practice, stochastic gradient descent is used as an umbrella term that encompasses both concepts.

Batch sizes are generally set to small numbers such as 1 or 2 to allow us to minimize computational cost. If we have access to a parallel machine (such as a GPU or a distributed computing system), we can increase the batch size to take advantage of the hardware. By running the algorithm in parallel, we could compute the gradient over a larger batch size in the same amount of time that the machine would take for a batch size of 1.

# Visualizing Stochastic Gradient Descent
