

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
np.random.seed(42)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
```


```python
# HIDDEN
tips = sns.load_dataset('tips')
tips['pcttip'] = tips['tip'] / tips['total_bill'] * 100
```


```python
# HIDDEN
def mse_cost(theta, y_vals):
    return np.mean((y_vals - theta) ** 2)

def grad_mse_cost(theta, y_vals):
    return -2 * np.mean(y_vals - theta)
```

## A Brief Review

We initially modeled the tips dataset by heuristically calculating the $\hat\theta$ that minimized the MSE cost function, $L(\hat\theta, y) = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat\theta)^2$. This involved taking the derivative of the MSE, setting it to zero, and solving for $\hat\theta$. We found that the MSE-minimizing $\hat\theta$ value was simply the mean of the $y$ values in our dataset.

But for more complicated models and cost functions, there may not be simple algebraic expressions that yield the cost-minimizing $\hat\theta$ values. Our solution was a new method called gradient descent, which iteratively updates $\hat\theta$ until it doesn't change between iterations. To complete an iteration of gradient descent, we calculate the following:

$$
\hat\theta_{t+1} = \hat\theta_t - \alpha \cdot \nabla_{\hat\theta} L(\hat\theta, y)
$$

## Limitations of Gradient Descent

Often, we calculate $\nabla_{\hat\theta}L(\hat\theta, y)$ using the average gradient of the loss function $l(\hat\theta, y_i)$ across the entire dataset as a single batch. For this reason, this gradient update rule is often referred to as **batch gradient descent**.

For example, the gradient of the MSE cost first requires us to find the gradient of the squared loss, $\nabla_{\hat\theta} l(\hat\theta, y_i) = -2 (y_i - \hat\theta)$, for each of the $n$ observations in our dataset. We then compute the average of these results. This is shown below:

$$
\begin{align}
\nabla_{\hat\theta} L(\hat\theta, y) &= -\frac{2}{n} \sum_{i=1}^{n}(y_i - \hat\theta) \\
&= \frac{1}{n} \sum_{i=1}^{n}-2(y_i - \hat\theta) \\
&= \frac{1}{n} \sum_{i=1}^{n} \nabla_{\hat\theta} l(\hat\theta, y_i)
\end{align}
$$



This process is then repeated at each iteration $t$ of the gradient descent algorithm. For large $n$, this can become a computationally expensive problem to solve.

## Stochastic Gradient Descent

To circumvent the difficulty of computing a gradient across the entire training set, **stochastic gradient descent** approximates the overall gradient using a single randomly chosen data point. Since the observation is chosen randomly, we expect that using the gradient at each individual observation will eventually converge the algorithm to the same parameters as batch gradient descent. The general update formula for stochastic gradient descent is below, where $l(\hat\theta, y_i)$ is the loss function for a single data point:

$$
\hat\theta_{t+1} = \hat\theta_t - \alpha \nabla_\hat\theta l(\hat\theta, y_i)
$$

Returning back to our example using the MSE, we approximate the gradient of the mean squared error using the gradient of the squared loss of one data point. 

$$
\begin{align}
\nabla_{\hat\theta}L(\hat\theta, y) &\approx \nabla_{\hat\theta} l(\hat\theta, y_i) \\
&= -2(y_i - \hat\theta)
\end{align}
$$

Stochastic gradient descent relies on the random selection of individual observations. This is statistically founded, since the randomness means that $E[\nabla_{\hat\theta}l(\hat\theta, y_i)] = \nabla_{\hat\theta}L(\hat\theta, y)$. In practice, we choose these random data points by shuffling the training data and iteratively selecting from the shuffled data. An iteration through the shuffled data is called an **epoch**; at the end of every epoch, we re-shuffle the data to ensure that the model sees examples from various classes in an arbitrary order.

### Visualizing Stochastic Gradient Descent

Below are visual examples of loss minimization in batch gradient descent and stochastic gradient descent.

![](gd.png)
![](sgd.png)

At each iteration of batch gradient descent, we move in the direction of the true gradient of the cost function, which is depicted by the ellipses. On the other hand, each iteration of stochastic gradient descent may not lead us in the direction of the true gradient; however, the $\hat\theta$ parameters eventually converge to the minima of the cost function. Surprisingly, stochastic gradient descent often converges more quickly than batch gradient descent in practice since stochastic gradient descent spends significantly less time updating model parameters.

### Defining a Function for Stochastic Gradient Descent

As we previously did for batch gradient descent, we would like to define a function that computes the stochastic gradient descent of the cost function. It will be similar to our `minimize` function, but we will need to implement the random selection of one observation at each iteration.


```python
def minimize_sgd(cost_fn, grad_cost_fn, dataset, alpha=0.2, progress=True):
    """
    Uses stochastic gradient descent to minimize cost_fn.
    Returns the minimizing value of theta once theta changes
    less than 0.001 between iterations.
    """
    NUM_OBS = len(dataset)
    theta = 0
    np.random.shuffle(dataset)
    while True:
        for i in range(0, NUM_OBS, 1):
            if progress:
                print(f'theta: {theta:.2f} | cost: {cost_fn(theta, dataset):.2f}')
            rand_obs = dataset[i]
            gradient = grad_cost_fn(theta, rand_obs)
            new_theta = theta - alpha * gradient
        
            if abs(new_theta - theta) < 0.001:
                return new_theta
        
            theta = new_theta
        np.random.shuffle(dataset)
```

## Mini-batch Gradient Descent

**Mini-batch gradient descent** strikes a balance between batch gradient descent and stochastic gradient descent by increasing the number of observations that we select at each iteration. In mini-batch gradient descent, we take a simple random sample of observations called a mini-batch. We use the average of the gradients of their loss functions to construct an estimate of the true gradient of the cross entropy cost. Since our sample is randomly selected, the expectation of this estimate is equal to the true gradient. This is illustrated in the approximation for the gradient of the cost function, where $\mathcal{B}$ is the mini-batch of data points that we randomly sample from the $n$ observations.

$$
\nabla_\hat\theta L(\hat\theta, y) \approx \frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}\nabla_{\hat\theta}l(\hat\theta, y_i)
$$

As with stochastic gradient descent, we perform mini-batch gradient descent by shuffling our training data and selecting mini-batches by iterating through the shuffled data. After each epoch, we re-shuffle our data and select new mini-batches.

While we have made the distinction between stochastic and mini-batch gradient descent, stochastic gradient descent is sometimes used as an umbrella term that encompasses the selection of a mini-batch of any size. 


### Selecting the Mini-Batch Size

Mini-batch gradient descent is most optimal when running on a Graphical Processing Unit (GPU) or on distributed systems. Since computations on these hardware machines can be executed in parallel, using a mini-batch can increase the accuracy of the gradient without increasing computation time. Depending on the memory of the GPU, the mini-batch size is often set between 10 and 100 observations. 

### Defining a Function for Mini-Batch Gradient Descent

A function for mini-batch gradient descent requires the ability to select a batch size. Below is a function that implements this feature.


```python
def minimize_mini_batch(cost_fn, grad_cost_fn, dataset, minibatch_size, alpha=0.2, progress=True):
    """
    Uses mini-batch gradient descent to minimize cost_fn.
    Returns the minimizing value of theta once theta changes
    less than 0.001 between iterations.
    """
    NUM_OBS = len(dataset)
    assert minibatch_size < NUM_OBS
    
    theta = 0
    np.random.shuffle(dataset)
    while True:
        for i in range(0, NUM_OBS, minibatch_size):
            if progress:
                print(f'theta: {theta:.2f} | cost: {cost_fn(theta, dataset):.2f}')
            
            mini_batch = dataset[i:i+minibatch_size]
            gradient = grad_cost_fn(theta, mini_batch)
            new_theta = theta - alpha * gradient
            
            if abs(new_theta - theta) < 0.1:
                return new_theta
            
            theta = new_theta
        np.random.shuffle(dataset)
```

## Summary

We use batch gradient descent to iteratively improve model parameters until the model achieves low cost. Since batch gradient descent is computationally intractable with large datasets, we often use stochastic gradient descent to fit models instead. With a GPU, mini-batch gradient descent can converge more quickly than stochastic gradient descent.

Although stochastic gradient descent typically takes more individual updates to converge than batch gradient descent, stochastic gradient descent often converges more quickly because it computes updates faster.
