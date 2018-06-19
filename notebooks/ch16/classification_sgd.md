

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

## Limitations of Gradient Descent

We have previously introduced gradient descent as a general-purpose algorithm used to find model parameters $ \hat\theta $ that minimize a specified cost function $ L(\hat\theta, X, y) $. To perform an iteration of gradient descent, we compute:

$$
\hat\theta_{t+1} = \hat\theta_t - \alpha \cdot \nabla_\hat\theta L(\hat\theta, X, y)
$$


Since $ L(\hat\theta, X, y) $ computes an average cost across an entire dataset, this gradient update equation computes a gradient using the average gradient across the entire dataset as a single batch. For this reason, this gradient update rule is often referred to as **batch gradient descent**.

For example, in logistic regression we use the cross entropy cost as our cost function. We replace  $\nabla_\hat\theta L(\hat\theta, X, y)$ with the gradient of the cross entropy cost, $-\frac{1}{n}\sum_{i=1}^n(y_i - \sigma_i)X_i$, to find the gradient descent algorithm specific to logistic regression. Letting $ \sigma_i = f_\hat\theta(X_i) = \sigma(X_i \cdot \hat \theta) $, this becomes:

$$
\hat\theta_{t+1} = \hat\theta_t - \alpha \cdot \left(- \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \sigma_i\right) X_i \right)
$$

To find the gradient of the cross entropy cost, we first compute the gradient of the cross entropy loss, $-(y_i - \sigma_i)X_i$, for all $n$ observations. Then we take the average over all observations to find the gradient of the cross entropy cost. This is then computed at each iteration $t$ of the gradient descent algorithm. For large $n$, this can become a computationally expensive problem to solve.

## Stochastic Gradient Descent

To circumvent the difficulty of computing a gradient across the entire training set, **stochastic gradient descent** approximates the overall gradient using a single randomly chosen data point. Since the observation is chosen randomly, we expect that using the gradient at each individual observation will eventually converge to the same parameters as batch gradient descent. The general update formula for stochastic gradient descent is below, where $l(\hat\theta, X, y)$ is the loss function for a single data point:

$$
\hat\theta_{t+1} = \hat\theta_t - \alpha \nabla_\hat\theta l(\hat\theta, X, y)
$$

Returning back to our example in logistic regression, we approximate the gradient of the cross entropy cost using the gradient of the cross entropy loss of one data point. This is shown below, with $ \sigma_i = f_\hat{\theta}(X_i) = \sigma(X_i \cdot \hat \theta) $.

$$
\begin{align}
\nabla_\hat\theta L(\hat\theta, X, y) &\approx \nabla_\hat\theta l(\hat\theta, X, y)\\
&= -(y_i - \sigma_i)X_i
\end{align}
$$

When we plug this approximation into the general formula for stochastic gradient descent, we find the stochastic gradient descent update formula for logistic regression.

$$
\begin{align}
\hat\theta_{t+1} &= \hat\theta_t - \alpha \nabla_\hat\theta l(\hat\theta, X, y) \\
&= \hat\theta_t + \alpha \cdot (y_i - \sigma_i)X_i
\end{align}
$$

In practice, we choose random data points by shuffling the training data and iteratively selecting from the shuffled data. An iteration through the shuffled data is called an **epoch**; at the end of every epoch, we re-shuffle the data to ensure that the model sees examples from various classes in an arbitrary order.

### Visualizing Stochastic Gradient Descent

Below are visual examples of loss minimization in batch gradient descent and stochastic gradient descent.

![](gd.png)
![](sgd.png)

At each iteration of batch gradient descent, we move in the direction of the true gradient of the cost function, which is depicted by the ellipses. On the other hand, each iteration of stochastic gradient descent may not lead us in the direction of the true gradient; however, the $\hat\theta$ parameters eventually converge to the minima of the cost function. Surprisingly, stochastic gradient descent often converges more quickly than batch gradient descent in practice since stochastic gradient descent spends significantly less time updating model parameters.

## Mini-batch Gradient Descent

**Mini-batch gradient descent** strikes a balance between batch gradient descent and stochastic gradient descent by increasing the number of observations that we select at each iteration. In mini-batch gradient descent, we take a simple random sample of observations called a mini-batch. We use the average of the gradients of their loss functions to construct an estimate of the true gradient of the cross entropy cost. Since our sample is randomly selected, the expectation of this estimate is equal to the true gradient. This is illustrated in the approximation for the cost function of logistic regression, where $\mathcal{B}$ is the mini-batch of data points that we randomly sample from the $n$ observations.

$$
\nabla_\hat\theta L(\hat\theta, X, y) \approx -\frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}(y_i - \sigma_i)X_i
$$

As with stochastic gradient descent, we perform mini-batch gradient descent by shuffling our training data and selecting mini-batches by iterating through the shuffled data. After each epoch, we re-shuffle our data and select new mini-batches.

While we have made the distinction between stochastic and mini-batch gradient descent, stochastic gradient descent is sometimes used as an umbrella term that encompasses the selection of a mini-batch of any size. 


### Selecting the Mini-Batch Size

Mini-batch gradient descent is most optimal when running on a Graphical Processing Unit (GPU) or on distributed systems. Since computations on these hardware machines can be executed in parallel, using a mini-batch can increase the accuracy of the gradient without increasing computation time. Depending on the memory of the GPU, the mini-batch size is often set between 10 and 100 observations. 

## Summary

We use batch gradient descent to iteratively improve model parameters until the model achieves low cost. Since batch gradient descent is computationally intractable with large datasets, we often use stochastic gradient descent to fit models instead. With a GPU, mini-batch gradient descent can converge more quickly than stochastic gradient descent.

Although stochastic gradient descent typically takes more individual updates to converge than batch gradient descent, stochastic gradient descent often converges more quickly because it computes updates faster.
