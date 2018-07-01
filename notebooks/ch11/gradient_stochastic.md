

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
def mse_loss(theta, y_vals):
    return np.mean((y_vals - theta) ** 2)

def grad_mse_loss(theta, y_vals):
    return -2 * np.mean(y_vals - theta)
```

## A Brief Review

We initially modeled the tips dataset by heuristically calculating the $\hat\theta$ that minimized the MSE loss function, $L(\hat\theta, y) = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat\theta)^2$. This involved taking the derivative of the MSE, setting it to zero, and solving for $\hat\theta$. We found that the MSE-minimizing $\hat\theta$ value was simply the mean of the $y$ values in our dataset.

But for more complicated models and loss functions, there may not be simple algebraic expressions that yield the loss-minimizing $\hat\theta$ values. Our solution was a new method called gradient descent, which iteratively updates $\hat\theta$ until it doesn't change between iterations. To complete an iteration of gradient descent, we calculate the following:

$$
\hat\theta_{t+1} = \hat\theta_t - \alpha \cdot \nabla_{\hat\theta} L(\hat\theta, y)
$$

In this equation:
- $\hat\theta_{t}$ is our current estimate of $\theta$ at the $t$th iteration
- $\alpha$ is the learning rate
- $\nabla_{\hat\theta} L(\hat\theta, y)$ is the gradient of the loss function
- We compute the next estimate $\hat\theta_{t+1}$ by subtracting the product of $\alpha$ and $\nabla_{\hat\theta} L(\hat\theta, y)$ computed at $\hat\theta_{t}$

## Limitations of Gradient Descent

Often, we calculate $\nabla_{\hat\theta}L(\hat\theta, y)$ using the average gradient of the loss function $l(\hat\theta, y_i)$ across the entire dataset as a single batch. For this reason, this gradient update rule is often referred to as **batch gradient descent**.

For example, the gradient of the MSE loss first requires us to find the gradient of the squared loss, $\nabla_{\hat\theta} l(\hat\theta, y_i) = -2 (y_i - \hat\theta)$, for each of the $n$ observations in our dataset. We then compute the average of these results. This is shown below:

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

Stochastic gradient descent relies on the random selection of individual observations. This is statistically founded since the randomness means that $E[\nabla_{\hat\theta}l(\hat\theta, y_i)] = \nabla_{\hat\theta}L(\hat\theta, y)$. In practice, we choose these random data points by shuffling the training data and iteratively selecting from the shuffled data. An iteration through all $n$ observations of the shuffled data is called an **epoch**; at the end of every epoch, we re-shuffle the data to ensure that the model sees data points in an arbitrary order.

### Visualizing Stochastic Gradient Descent

Below are visual examples of loss minimization in batch gradient descent and stochastic gradient descent.

![](gd.png)
![](sgd.png)

At each iteration of batch gradient descent, we move in the direction of the true gradient of the loss function, which is depicted by the ellipses. On the other hand, each iteration of stochastic gradient descent may not lead us in the direction of the true gradient; however, the $\hat\theta$ parameters eventually converge to the minima of the loss function. Although stochastic gradient descent typically takes more iterations to converge than batch gradient descent, it often converges more quickly because it spends significantly less time computing the update at each iteration.


```python
##### 4 TEMPORARY CELLS BELOW TO TEST OUT %%TIME FOR BATCH GRADIENT DESCENT.
```


```python
def minimize(loss_fn, grad_loss_fn, dataset, alpha=0.2, progress=True):
    '''
    Uses gradient descent to minimize loss_fn. Returns the minimizing value of
    theta once theta changes less than 0.001 between iterations.
    '''
    theta = 0
    while True:
        if progress:
            print(f'theta: {theta:.2f} | loss: {loss_fn(theta, dataset):.2f}')
        gradient = grad_loss_fn(theta, dataset)
        new_theta = theta - alpha * gradient
        
        if abs(new_theta - theta) < 0.001:
            return new_theta
        
        theta = new_theta
```


```python
# defining sets of sample_data and entire_data to test out gd, sgd, mbgd
sample_data = np.array([12.1, 12.8, 14.9, 16.3, 17.2])
entire_data = np.array(tips['pcttip'])
```


```python
# This yields the correct theta that the algorithm should converge to.
np.mean(entire_data)
```




    16.08025817225047




```python
%%time
minimize(mse_loss, grad_mse_loss, entire_data)
```

    theta: 0.00 | loss: 295.72
    theta: 6.43 | loss: 130.23
    theta: 10.29 | loss: 70.66
    theta: 12.61 | loss: 49.21
    theta: 14.00 | loss: 41.49
    theta: 14.83 | loss: 38.71
    theta: 15.33 | loss: 37.71
    theta: 15.63 | loss: 37.35
    theta: 15.81 | loss: 37.22
    theta: 15.92 | loss: 37.17
    theta: 15.98 | loss: 37.15
    theta: 16.02 | loss: 37.15
    theta: 16.05 | loss: 37.15
    theta: 16.06 | loss: 37.15
    theta: 16.07 | loss: 37.15
    theta: 16.07 | loss: 37.15
    theta: 16.08 | loss: 37.15
    theta: 16.08 | loss: 37.15
    theta: 16.08 | loss: 37.15
    CPU times: user 15.6 ms, sys: 0 ns, total: 15.6 ms
    Wall time: 1.75 ms





    16.07927830605656



### Defining a Function for Stochastic Gradient Descent

As we previously did for batch gradient descent, we would like to define a function that computes the stochastic gradient descent of the loss function. It will be similar to our `minimize` function, but we will need to implement the random selection of one observation at each iteration.


```python
def minimize_sgd(loss_fn, grad_loss_fn, dataset, alpha=0.2, progress=True):
    """
    Uses stochastic gradient descent to minimize loss_fn.
    Returns the minimizing value of theta once theta changes
    less than 0.001 between iterations.
    """
    NUM_OBS = len(dataset)
    theta = 0
    np.random.shuffle(dataset)
    while True:
        for i in range(0, NUM_OBS, 1):
            if progress:
                print(f'theta: {theta:.2f} | loss: {loss_fn(theta, dataset):.2f}')
            rand_obs = dataset[i]
            gradient = grad_loss_fn(theta, rand_obs)
            new_theta = theta - alpha * gradient
        
            if abs(new_theta - theta) < 0.1:
                return new_theta
        
            theta = new_theta
        np.random.shuffle(dataset)
```


```python
### TEMPORARY CELL BELOW; minimize_sgd does not converge at any alpha values that I have tried.
# I have tried to adjust alpha values and/or the threshold above ("if abs(new_theta - theta) < _____") 
# until I could find one that works (which I have not yet)
```


```python
%%time
minimize_sgd(mse_loss, grad_mse_loss, entire_data, alpha=0.1)
```

    theta: 0.00 | loss: 295.72
    theta: 3.21 | loss: 202.82
    theta: 7.13 | loss: 117.29
    theta: 7.33 | loss: 113.68
    theta: 9.85 | loss: 75.94
    theta: 10.96 | loss: 63.38
    CPU times: user 0 ns, sys: 0 ns, total: 0 ns
    Wall time: 2.23 ms





    11.002413901512812



## Mini-batch Gradient Descent

**Mini-batch gradient descent** strikes a balance between batch gradient descent and stochastic gradient descent by increasing the number of observations that we select at each iteration. In mini-batch gradient descent, we take a simple random sample of observations called a mini-batch. We use the average of the gradients of their loss functions to construct an estimate of the true gradient of the cross entropy loss. Since our sample is randomly selected, the expectation of this estimate is equal to the true gradient. This is illustrated in the approximation for the gradient of the loss function, where $\mathcal{B}$ is the mini-batch of data points that we randomly sample from the $n$ observations.

$$
\nabla_\hat\theta L(\hat\theta, y) \approx \frac{1}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}\nabla_{\hat\theta}l(\hat\theta, y_i)
$$

As with stochastic gradient descent, we perform mini-batch gradient descent by shuffling our training data and selecting mini-batches by iterating through the shuffled data. After each epoch, we re-shuffle our data and select new mini-batches.

While we have made the distinction between stochastic and mini-batch gradient descent in this textbook, stochastic gradient descent is sometimes used as an umbrella term that encompasses the selection of a mini-batch of any size. 


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


```python
### TEMPORARY CELL BELOW; minimize_mini_batch does not converge at any values of alpha/thresholds that I have tried
```


```python
%%time
minimize_mini_batch(mse_loss, grad_mse_loss, entire_data, 20, alpha=0.08)
```

    theta: 0.00 | cost: 295.72
    theta: 2.76 | cost: 214.57
    theta: 4.70 | cost: 166.59
    theta: 6.49 | cost: 129.11
    theta: 7.87 | cost: 104.55
    theta: 9.32 | cost: 82.84
    theta: 10.52 | cost: 68.01
    theta: 11.33 | cost: 59.75
    theta: 12.14 | cost: 52.64
    theta: 12.76 | cost: 48.15
    theta: 13.17 | cost: 45.59
    theta: 13.48 | cost: 43.89
    theta: 14.03 | cost: 41.33
    theta: 14.91 | cost: 38.51
    CPU times: user 0 ns, sys: 0 ns, total: 0 ns
    Wall time: 2.38 ms





    14.837532029516163



## Summary

We use batch gradient descent to iteratively improve model parameters until the model achieves minimal loss. Since batch gradient descent is computationally intractable with large datasets, we often use stochastic gradient descent to fit models instead. Furthermore, when using a GPU, mini-batch gradient descent can converge more quickly than stochastic gradient descent for the same computational cost. For large datasets, stochastic gradient descent and mini-batch gradient descent are often preferred to batch gradient descent for their faster computation times.
